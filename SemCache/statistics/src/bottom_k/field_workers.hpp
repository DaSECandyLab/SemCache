
class FieldWorkers {
public:
  explicit FieldWorkers(std::size_t limit) : limit_(limit) {}
  ~FieldWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_all();
    for (auto &thread : threads_)
      thread.join();
  }
  FieldWorkers(const FieldWorkers &) = delete;
  FieldWorkers &operator=(const FieldWorkers &) = delete;
  std::size_t width() const { return threads_.size() + 1; }
  void Run(std::size_t count, const std::function<void(std::size_t)> &operation,
           std::vector<std::size_t> order = {}) {
    if (limit_ < 2 || count < 2) {
      for (std::size_t f = 0; f < count; ++f)
        operation(f);
      return;
    }
    if (order.empty()) {
      order.resize(count);
      std::iota(order.begin(), order.end(), 0);
    }
    while (threads_.size() + 1 < std::min(limit_, count))
      threads_.emplace_back([this] { Worker(); });
    {
      std::lock_guard<std::mutex> lock(mutex_);
      order_ = std::move(order);
      errors_.assign(count, nullptr);
      operation_ = operation;
      next_ = 0;
      pending_ = threads_.size();
      ++generation_;
    }
    ready_.notify_all();
    Execute();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      finished_.wait(lock, [&] { return pending_ == 0; });
      operation_ = {};
    }
    for (const auto &error : errors_)
      if (error)
        std::rethrow_exception(error);
  }

private:
  void Execute() noexcept {
    while (true) {
      const auto index = next_.fetch_add(1, std::memory_order_relaxed);
      if (index >= order_.size())
        return;
      const auto field = order_[index];
      try {
        operation_(field);
      } catch (...) {
        errors_[field] = std::current_exception();
      }
    }
  }
  void Worker() {
    std::size_t generation = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      ready_.wait(lock, [&] { return stop_ || generation != generation_; });
      if (stop_)
        return;
      generation = generation_;
      lock.unlock();
      Execute();
      lock.lock();
      if (!--pending_)
        finished_.notify_one();
    }
  }
  std::size_t limit_, generation_ = 0, pending_ = 0;
  bool stop_ = false;
  std::vector<std::thread> threads_;
  std::vector<std::size_t> order_;
  std::vector<std::exception_ptr> errors_;
  std::function<void(std::size_t)> operation_;
  std::atomic<std::size_t> next_{0};
  std::mutex mutex_;
  std::condition_variable ready_, finished_;
};

// The existing fetch ABI only promises buffers until the next callback. Copy
// under the callback lock before another field can invalidate those buffers.
struct FieldBatch {
  AdaptiveFetch fetch;
  void *context;
  std::mutex *mutex;
  std::vector<const char *> data;
  std::vector<std::size_t> sizes;
  std::vector<std::string> owned;
  void Read(std::size_t field, const std::vector<std::size_t> &ids) {
    std::unique_lock<std::mutex> lock;
    if (mutex)
      lock = std::unique_lock<std::mutex>(*mutex);
    data.resize(ids.size());
    sizes.resize(ids.size());
    if (fetch(field, ids.data(), ids.size(), data.data(), sizes.data(),
              context))
      throw std::runtime_error("Adaptive source callback failed");
    if (mutex) {
      owned.resize(ids.size());
      for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!data[i] && sizes[i])
          throw std::invalid_argument("Null string pointer");
        owned[i].assign(data[i] ? data[i] : "", sizes[i]);
        data[i] = owned[i].data();
      }
    }
  }
};
