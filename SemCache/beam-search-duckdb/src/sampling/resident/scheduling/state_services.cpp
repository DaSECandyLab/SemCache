#include "../detail.hpp"

namespace semcache::sampling {
thread_local NativeSampler::Impl::StateActor *NativeSampler::Impl::state_actor =
    nullptr;
NativeSampler::Impl::StateActor::StateActor(Impl &o, OrderedTasks::Task &t,
                                            StateWorker &w)
    : owner(o), task(t), worker(w), saved(o.backend) {
  owner.backend = worker.backend;
  owner.state_actor = this;
  task.hooks([this] { leave(); }, [this] { enter(); });
}
void NativeSampler::Impl::StateActor::leave() {
  owner.backend = saved;
  owner.state_actor = nullptr;
}
void NativeSampler::Impl::StateActor::enter() {
  saved = owner.backend;
  owner.backend = worker.backend;
  owner.state_actor = this;
}
NativeSampler::Impl::StateActor::~StateActor() {
  task.turn();
  task.hooks();
  owner.backend = saved;
  owner.state_actor = nullptr;
}
void NativeSampler::Impl::state_turn() {
  if (state_actor)
    state_actor->task.turn();
}
Json NativeSampler::Impl::state_backend_call(
    const std::function<Json()> &operation,
    const std::function<void()> &prepare) {
  if (!state_actor)
    return sql_pipeline.run(operation, prepare);
  auto &actor = *state_actor;
  if (actor.task.failed())
    throw std::runtime_error("Sampling wave stopped after an earlier failure");
  if (!prepare)
    return actor.task.outside(operation);
  auto pending = actor.worker.pipeline.submit(operation);
  std::exception_ptr preparation_error;
  try {
    prepare();
  } catch (...) {
    preparation_error = std::current_exception();
  }
  auto result = actor.task.outside([&] { return pending.get(); });
  if (preparation_error)
    std::rethrow_exception(preparation_error);
  return result;
}
void NativeSampler::Impl::run_state_services(
    std::size_t count, const std::function<void(std::size_t)> &operation) {
  const auto original = backend;
  if (state_slots.size() < state_workers->size())
    state_slots.resize(state_workers->size());
  state_workers->run(
      count,
      [&](OrderedTasks::Task &task) {
        StateActor actor(*this, task, *state_slots[task.slot]);
        operation(task.index);
      },
      [&](std::size_t slot) {
        if (!state_slots[slot]) {
          auto created = std::make_unique<StateWorker>();
          created->backend = slot == 0 ? original : original->fork_semantic();
          state_slots[slot] = std::move(created);
        }
      });
  if (pending_selection || !physical_claims.empty() ||
      !selection_claims.empty() || !claim_waiters.empty())
    throw std::logic_error("Sampling wave retained an in-flight reservation");
}
NativeSampler::Impl::StateClaim::StateClaim(
    Impl &o, std::map<std::string, std::size_t> &bank,
    std::vector<std::string> requested)
    : owner(o), claims(bank), keys(std::move(requested)) {
  if (!owner.state_actor)
    return;
  auto &task = owner.state_actor->task;
  ticket = task.index;
  for (const auto &key : keys)
    owner.claim_waiters[{&claims, key}].insert(*ticket);
  task.wait([&] {
    for (const auto &key : keys) {
      const auto found = claims.find(key);
      if (found != claims.end() && found->second != *ticket)
        return false;
      // Waking later is not permission to overtake an earlier waiter: the
      // later owner would wait for that earlier task's ordered publication.
      if (*owner.claim_waiters.at({&claims, key}).begin() != *ticket)
        return false;
    }
    return true;
  });
  for (const auto &key : keys) {
    auto queued = owner.claim_waiters.find({&claims, key});
    queued->second.erase(*ticket);
    if (queued->second.empty())
      owner.claim_waiters.erase(queued);
    claims[key] = *ticket;
  }
}
NativeSampler::Impl::StateClaim::~StateClaim() { release(); }
void NativeSampler::Impl::StateClaim::release() {
  if (!ticket)
    return;
  for (const auto &key : keys)
    claims.erase(key);
  owner.state_actor->task.notify();
  ticket.reset();
}
}
