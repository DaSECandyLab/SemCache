#pragma once
#include "../populations.hpp"
#include "../random.hpp"
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace semcache::sampling::resident_detail {
inline bool join_operator(const Json &op, std::size_t input_count) {
  auto kind = op.value("kind", op.value("type", std::string()));
  std::transform(kind.begin(), kind.end(), kind.begin(), ::toupper);
  const bool semantic = op.value("semantic", kind.rfind("SEMANTIC", 0) == 0);
  return kind.find("JOIN") != std::string::npos || kind == "CROSS_PRODUCT" ||
         (!semantic && kind == "FILTER" && input_count == 2 &&
          op.contains("candidate_sql") && op.at("candidate_sql").is_string() &&
          !op.at("candidate_sql").get_ref<const std::string &>().empty());
}

// Retain coordinates, not expanded Cartesian rows. Replenishment keeps the
// previously published identities and appends unseen coordinates.
class JoinOutputStream {
public:
  // An immutable view of one successful prefix. Native evidence guarantees
  // that prefixes of the same owner have nested identity sets, even when
  // novelty-first delivery changes their order. Callbacks own that prefix.
  struct GeneratedPopulation {
    std::shared_ptr<const void> owner;
    std::size_t count = 0;
    std::function<const std::string &(std::size_t)> identity;
    std::function<std::optional<std::size_t>(const std::string &)> position;
    bool nested = false;
  };

private:
  using Coordinate = std::vector<std::string>;
  struct Entry {
    Coordinate coordinate;
    std::shared_ptr<const Json> row;
  };
  struct State {
    std::vector<Entry> selected;
    // Population membership/equality never controls delivery order. Avoid
    // rebuilding a sorted tree for every growing successful probe prefix.
    std::vector<std::unordered_set<std::string>> population;
    std::shared_ptr<const GeneratedPopulation> generated;
    bool approximate = false;
    // An initial census borrows its immutable representation until expansion.
    std::shared_ptr<const Json> census;
    std::shared_ptr<const void> evidence;
    std::shared_ptr<const Json> source;
    std::shared_ptr<const Json> publication;
    std::map<std::string, double> source_inclusions;
    std::size_t target = 0;
    std::uint64_t seed = 0;
  };
  std::shared_ptr<const State> state_ = std::make_shared<const State>();
  std::size_t population_materializations_ = 0;

  static bool same_population(const State &previous,
                              const GeneratedPopulation &current) {
    if (previous.generated) {
      const auto &old = *previous.generated;
      if (old.count != current.count)
        return false;
      if (old.nested && current.nested && old.owner == current.owner)
        return true;
      for (std::size_t i = 0; i < old.count; ++i)
        if (!current.position(old.identity(i)))
          return false;
      return true;
    }
    if (previous.population.size() != 1 ||
        previous.population[0].size() != current.count)
      return false;
    for (const auto &id : previous.population[0])
      if (!current.position(id))
        return false;
    return true;
  }

  static bool
  same_population(const State &previous,
                  const std::vector<std::unordered_set<std::string>> &current) {
    if (!previous.generated)
      return previous.population == current;
    if (current.size() != 1 || current[0].size() != previous.generated->count)
      return false;
    for (std::size_t i = 0; i < previous.generated->count; ++i)
      if (!current[0].count(previous.generated->identity(i)))
        return false;
    return true;
  }

  std::shared_ptr<const GeneratedPopulation>
  materialize_population(const std::vector<std::string> &ids) {
    auto ordered = std::make_shared<const std::vector<std::string>>(ids);
    auto index =
        std::make_shared<std::unordered_map<std::string, std::size_t>>();
    index->reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
      if (!index->emplace(ids[i], i).second)
        throw std::invalid_argument(
            "Join output requires stable unique row identities");
    ++population_materializations_;
    return std::make_shared<const GeneratedPopulation>(GeneratedPopulation{
        ordered, ids.size(),
        [ordered](std::size_t i) -> const std::string & {
          return ordered->at(i);
        },
        [index](const std::string &id) -> std::optional<std::size_t> {
          const auto found = index->find(id);
          return found == index->end()
                     ? std::nullopt
                     : std::optional<std::size_t>(found->second);
        },
        false});
  }

  static void leaves(const Json &rows, std::vector<const Json *> &out) {
    if (product(rows)) {
      for (const auto &side : rows.at("product"))
        leaves(side, out);
    } else
      out.push_back(&rows);
  }
  static Json restore(const Json &rows, const std::vector<std::size_t> &indices,
                      std::size_t &side) {
    if (!product(rows))
      return rows.at(indices.at(side++));
    const auto &parts = rows.at("product");
    auto left = restore(parts[0], indices, side);
    auto right = restore(parts[1], indices, side);
    return combine(left, right);
  }

public:
  bool empty() const {
    return state_->selected.empty() &&
           (!state_->census || !row_count(*state_->census));
  }
  void remember_census(std::shared_ptr<const Json> rows) {
    auto next = std::make_shared<State>();
    next->census = std::move(rows);
    state_ = std::move(next);
  }
  std::shared_ptr<const Json> publication(std::shared_ptr<const Json> rows,
                                          std::size_t requested,
                                          std::uint64_t seed) {
    if (state_->source == rows && state_->publication &&
        state_->target == requested && state_->seed == seed)
      return state_->publication;
    // A complete flat publication with the same delivered order already owns
    // the current payloads/weights. No restore/copy/reweight pass is needed.
    // At most one unseen row is allowed: with two or more, the normal cursor
    // randomizes their delivery order, which this shortcut must not change.
    const auto census = state_->census && state_->census->is_array()
                            ? state_->census.get()
                            : nullptr;
    const auto delivered = census ? census->size() : state_->selected.size();
    bool identical_order =
        rows->is_array() && rows->size() <= requested &&
        rows->size() >= delivered && rows->size() - delivered <= 1 &&
        !state_->generated &&
        (census || (!state_->census && state_->population.size() == 1));
    for (std::size_t i = 0; identical_order && i < delivered; ++i) {
      const auto &old = census ? census->at(i) : *state_->selected[i].row;
      identical_order = old.at("id") == rows->at(i).at("id");
    }
    if (identical_order) {
      std::unordered_set<std::string> ids;
      for (const auto &row : *rows)
        if (!ids.insert(row.at("id").get<std::string>()).second)
          throw std::invalid_argument(
              "Join output requires stable unique row identities");
      auto next = std::make_shared<State>();
      next->census = rows;
      next->source = rows;
      next->publication = rows;
      next->target = requested;
      next->seed = seed;
      state_ = std::move(next);
      return rows;
    }
    auto result = std::make_shared<const Json>(take(*rows, requested, seed));
    auto next = std::make_shared<State>(*state_);
    next->source = std::move(rows);
    next->publication = result;
    next->target = requested;
    next->seed = seed;
    state_ = std::move(next);
    return result;
  }
  std::shared_ptr<const Json>
  generated_publication(const std::vector<std::string> &ids,
                        const std::function<Json(std::size_t)> &restore,
                        std::size_t requested, std::uint64_t seed,
                        std::shared_ptr<const void> evidence) {
    if (state_->evidence == evidence && state_->publication &&
        state_->target == requested && state_->seed == seed)
      return state_->publication;
    return generated_publication(materialize_population(ids), restore,
                                 requested, seed, std::move(evidence));
  }
  std::shared_ptr<const Json>
  generated_publication(std::shared_ptr<const GeneratedPopulation> population,
                        const std::function<Json(std::size_t)> &restore,
                        std::size_t requested, std::uint64_t seed,
                        std::shared_ptr<const void> evidence) {
    if (state_->evidence == evidence && state_->publication &&
        state_->target == requested && state_->seed == seed)
      return state_->publication;
    auto rows = std::make_shared<const Json>(
        take_generated(std::move(population), restore, requested, seed));
    auto next = std::make_shared<State>(*state_);
    next->evidence = std::move(evidence);
    next->publication = rows;
    next->target = requested;
    next->seed = seed;
    state_ = std::move(next);
    return rows;
  }
  bool approximate() const { return state_->approximate; }
  std::size_t population_materializations() const {
    return population_materializations_;
  }
  Json take_generated(const std::vector<std::string> &ids,
                      const std::function<Json(std::size_t)> &restore,
                      std::size_t requested, std::uint64_t seed) {
    return take_generated(materialize_population(ids), restore, requested,
                          seed);
  }
  Json take_generated(std::shared_ptr<const GeneratedPopulation> population,
                      const std::function<Json(std::size_t)> &restore,
                      std::size_t requested, std::uint64_t seed) {
    auto previous = state_;
    if (state_->census) {
      // Materialize only at the census -> generated transition. The borrowed
      // publication is already delivered, including its original row order.
      auto retained = std::make_shared<State>();
      retained->population.emplace_back();
      const auto &census = *state_->census;
      for (std::size_t i = 0; i < row_count(census); ++i) {
        auto row = std::make_shared<const Json>(row_at(census, i));
        const auto id = row->at("id").get<std::string>();
        if (!retained->population[0].insert(id).second)
          throw std::invalid_argument(
              "Join output requires stable unique row identities");
        retained->selected.push_back({{id}, std::move(row)});
      }
      previous = std::move(retained);
    }
    auto next = std::make_shared<State>();
    next->generated = std::move(population);
    const auto &current = *next->generated;
    next->approximate =
        previous->approximate ||
        ((previous->generated || !previous->population.empty()) &&
         !same_population(*previous, current));
    std::set<std::string> seen;
    for (const auto &entry : previous->selected) {
      if (next->selected.size() >= requested)
        break;
      auto retained = entry;
      const auto id = entry.row->at("id").get<std::string>();
      auto found = current.position(id);
      if (found) {
        retained.coordinate = {id};
        retained.row = std::make_shared<const Json>(restore(*found));
      } else
        next->approximate = true;
      next->selected.push_back(std::move(retained));
      seen.insert(id);
    }
    if (previous->selected.empty() && current.count <= requested) {
      // A first untruncated publication preserves its input order but still
      // records delivery, so later truncation cannot replace those identities.
      for (std::size_t i = 0; i < current.count; ++i) {
        next->selected.push_back(
            {{current.identity(i)}, std::make_shared<const Json>(restore(i))});
        seen.insert(current.identity(i));
      }
    }
    Random random(seed);
    std::unordered_map<std::size_t, std::size_t> swaps;
    for (std::size_t cursor = 0;
         next->selected.size() < requested && cursor < current.count;
         ++cursor) {
      const auto selected = cursor + random.below(current.count - cursor);
      auto read = [&](std::size_t i) {
        auto found = swaps.find(i);
        return found == swaps.end() ? i : found->second;
      };
      const auto position = read(selected), replacement = read(cursor);
      swaps.erase(cursor);
      if (selected != cursor)
        swaps[selected] = replacement;
      if (seen.insert(current.identity(position)).second)
        next->selected.push_back(
            {{current.identity(position)},
             std::make_shared<const Json>(restore(position))});
    }
    // Full current coverage removes earlier subsampling uncertainty. Retained
    // matches absent from this publication still require the fallback marker.
    if (next->selected.size() == current.count &&
        std::all_of(next->selected.begin(), next->selected.end(),
                    [&](const Entry &entry) {
                      return bool(current.position(
                          entry.row->at("id").get<std::string>()));
                    }))
      next->approximate = false;
    state_ = std::move(next);
    return publish(current.count);
  }
  Json take(const Json &rows, std::size_t requested, std::uint64_t seed) {
    auto previous = state_;
    if (state_->census) {
      auto retained = std::make_shared<State>();
      const auto &census = *state_->census;
      std::vector<const Json *> original;
      leaves(census, original);
      retained->population.resize(original.size());
      for (std::size_t side = 0; side < original.size(); ++side)
        for (const auto &row : *original[side])
          if (!retained->population[side]
                   .insert(row.at("id").get<std::string>())
                   .second)
            throw std::invalid_argument(
                "Join output requires stable unique row identities");
      for (std::size_t i = 0; i < row_count(census); ++i) {
        Coordinate coordinate(original.size());
        auto position = i;
        for (std::size_t side = original.size(); side--;) {
          const auto n = original[side]->size();
          coordinate[side] =
              original[side]->at(position % n).at("id").get<std::string>();
          position /= n;
        }
        retained->selected.push_back(
            {std::move(coordinate),
             std::make_shared<const Json>(row_at(census, i))});
      }
      previous = std::move(retained);
    }
    std::vector<const Json *> banks;
    leaves(rows, banks);
    std::vector<std::map<std::string, std::size_t>> indexes(banks.size());
    std::vector<std::unordered_set<std::string>> population(banks.size());
    for (std::size_t side = 0; side < banks.size(); ++side)
      for (std::size_t i = 0; i < banks[side]->size(); ++i) {
        const auto id = banks[side]->at(i).at("id").get<std::string>();
        if (!indexes[side].emplace(id, i).second)
          throw std::invalid_argument(
              "Join output requires stable unique row identities");
        population[side].insert(id);
      }
    auto next = std::make_shared<State>();
    next->approximate =
        previous->approximate ||
        ((previous->generated || !previous->population.empty()) &&
         !same_population(*previous, population));
    next->population = std::move(population);
    const auto count = row_count(rows);
    std::set<Coordinate> seen;
    std::set<std::string> identities;
    // Only the published batch owns fallback rows. Probe resampling cannot
    // erase proven matches; raw populations and Cartesian products stay lazy.
    for (const auto &entry : previous->selected) {
      if (next->selected.size() >= requested)
        break;
      auto retained = entry;
      bool present = entry.coordinate.size() == banks.size();
      std::vector<std::size_t> indices;
      for (std::size_t side = 0; present && side < banks.size(); ++side) {
        auto found = indexes[side].find(entry.coordinate[side]);
        present = found != indexes[side].end();
        if (present)
          indices.push_back(found->second);
      }
      if (present && previous->source.get() != &rows) {
        std::size_t side = 0;
        retained.row =
            std::make_shared<const Json>(restore(rows, indices, side));
      } else if (!present)
        next->approximate = true;
      next->selected.push_back(std::move(retained));
      seen.insert(entry.coordinate);
      identities.insert(entry.row->at("id").get<std::string>());
    }
    Random random(seed);
    std::unordered_map<std::size_t, std::size_t> swaps;
    // Lazy Fisher-Yates avoids an array of count Cartesian coordinates.
    for (std::size_t cursor = 0;
         next->selected.size() < requested && cursor < count; ++cursor) {
      const auto position = cursor + random.below(count - cursor);
      auto read = [&](std::size_t i) {
        auto found = swaps.find(i);
        return found == swaps.end() ? i : found->second;
      };
      auto index = read(position), replacement = read(cursor);
      swaps.erase(cursor);
      if (position != cursor)
        swaps[position] = replacement;
      Coordinate coordinate(banks.size());
      for (std::size_t side = banks.size(); side--;) {
        const auto n = banks[side]->size();
        coordinate[side] =
            banks[side]->at(index % n).at("id").get<std::string>();
        index /= n;
      }
      if (seen.insert(coordinate).second) {
        std::vector<std::size_t> indices;
        for (std::size_t side = 0; side < banks.size(); ++side)
          indices.push_back(indexes[side].at(coordinate[side]));
        std::size_t side = 0;
        auto row = std::make_shared<const Json>(restore(rows, indices, side));
        if (identities.insert(row->at("id").get<std::string>()).second)
          next->selected.push_back({std::move(coordinate), std::move(row)});
      }
    }
    if (next->selected.size() == count &&
        std::all_of(next->selected.begin(), next->selected.end(),
                    [&](const Entry &entry) {
                      if (entry.coordinate.size() != indexes.size())
                        return false;
                      for (std::size_t side = 0; side < indexes.size(); ++side)
                        if (!indexes[side].count(entry.coordinate[side]))
                          return false;
                      return true;
                    }))
      next->approximate = false;
    if (!next->approximate && next->selected.size() < count) {
      // Work on the factor leaves, never expand the Cartesian product. The
      // probability concerns a distinct ancestor appearing at least once.
      std::set<std::string> needed;
      for (const auto &entry : next->selected)
        for (const auto &origin : entry.row->at("lineage"))
          needed.insert(canonical(origin));
      std::map<std::string, std::size_t> absent;
      for (const auto &key : needed)
        absent[key] = 1;
      for (const auto *bank : banks) {
        std::map<std::string, std::size_t> copies;
        for (const auto &row : *bank) {
          std::set<std::string> seen;
          for (const auto &origin : row.at("lineage")) {
            const auto key = canonical(origin);
            if (needed.count(key) && seen.insert(key).second)
              ++copies[key];
          }
        }
        for (auto &[key, missing] : absent)
          missing *= bank->size() - copies[key];
      }
      for (const auto &[key, missing] : absent)
        next->source_inclusions[key] =
            source_inclusion(count, count - missing, next->selected.size());
    }
    state_ = std::move(next);
    return publish(count);
  }

private:
  Json publish(std::size_t count) const {
    Json result = Json::array();
    result.get_ref<Json::array_t &>().reserve(state_->selected.size());
    const bool limited = state_->selected.size() < count;
    const auto probability =
        limited ? double(state_->selected.size()) / count : 1.;
    for (const auto &entry : state_->selected) {
      auto row = *entry.row;
      if (limited || state_->approximate) {
        const bool weighted = row.contains("_sample_weight_factor");
        sample_factor(row, sample_factor(row) / probability);
        Json probabilities = Json::object();
        const auto previous =
            row.value("_sample_origin_probabilities", Json::object());
        for (const auto &origin : row.at("lineage")) {
          auto key = canonical(origin);
          const auto prior =
              previous.value(key, weighted ? Json(nullptr) : Json(1.));
          const auto included = state_->source_inclusions.find(key);
          probabilities[key] =
              !state_->approximate && finite_number(prior) &&
                      included != state_->source_inclusions.end()
                  ? Json(prior.get<double>() * included->second)
                  : Json(nullptr);
        }
        row["_sample_origin_probabilities"] = std::move(probabilities);
        row["_sample_match_incomplete"] = true;
        // Multiple matches share ancestors; a nominal expansion does not
        // establish a precise distinct-source inclusion probability.
        row["_sample_origin_approximate"] = true;
      }
      result.push_back(std::move(row));
    }
    return result;
  }
};
}
