#include "llm/http/http_client_internal.hpp"
#include <random>
#include <vector>

namespace semantic {

using namespace http_detail;

LLMResponse HttpLLMClient::InferSemanticCached(const LLMRequest &request, const std::function<void()> &admitted) {
	auto reuse = [&](const LLMResponse &source) {
		ValidateStructuredPayload(request, source.raw_text);
		LLMResponse cached;
		cached.raw_text = source.raw_text;
		cached.cache_result_confirmed = true;
		cached.result_cache_hit = true;
		cached.input_tokens = source.input_tokens;
		cached.submit_ns = cached.response_ns = 0;
		if (on_cache_hit)
			on_cache_hit(request);
		return cached;
	};
	if (request.cached_response && request.cached_response->cache_result_confirmed)
		return reuse(*request.cached_response);
	const auto contract = CacheContract(request);
	auto flight_key = request.result_cache_key;
	// Clients sharing storage may choose different verification policies.
	AppendKeyPart(flight_key, verify_on_second_use ? "verify" : "reuse");
	LLMResponse provisional;
	bool have_provisional = false;
	for (;;) {
		LLMResponse stored;
		bool confirmed = false;
		if (LookupCached(request, stored, confirmed)) {
			if (confirmed)
				return reuse(stored);
			provisional = std::move(stored);
			have_provisional = true;
		}
		std::shared_ptr<std::promise<LLMResponse>> owner;
		std::shared_future<LLMResponse> future;
		{
			std::lock_guard<std::mutex> guard(cache_flights->mutex);
			auto found = cache_flights->requests.find(flight_key);
			if (found != cache_flights->requests.end()) {
				future = found->second;
			} else {
				owner = std::make_shared<std::promise<LLMResponse>>();
				future = owner->get_future().share();
				cache_flights->requests.emplace(flight_key, future);
			}
		}
		if (!owner) {
			auto completed = future.get();
			ValidateStructuredPayload(request, completed.raw_text);
			if (completed.cache_result_confirmed)
				return reuse(completed);
			// A cold owner has supplied only the first observation. A follower
			// must acquire the next phase and make an independent model call.
			provisional = std::move(completed);
			have_provisional = true;
			continue;
		}
		auto finish = [&]() {
			std::lock_guard<std::mutex> guard(cache_flights->mutex);
			cache_flights->requests.erase(flight_key);
		};
		try {
			// Close publication/removal races before admitting another call.
			if (LookupCached(request, stored, confirmed, have_provisional)) {
				if (confirmed) {
					auto response = reuse(stored);
					finish();
					owner->set_value(response);
					return response;
				}
				provisional = std::move(stored);
				have_provisional = true;
			}
			auto response = InferHTTP(request, admitted);
			response.cache_result_confirmed = !verify_on_second_use;
			auto latest = llm_detail::ParseJSON(response.raw_text);
			Json winner = latest;
			Json summary;
			if (have_provisional) {
				struct Candidate {
					Json value;
					size_t votes;
				};
				std::vector<Candidate> candidates;
				size_t observations = 0;
				auto vote = [&](const Json &value) {
					++observations;
					for (auto &candidate : candidates) {
						if (candidate.value == value) {
							++candidate.votes;
							return;
						}
					}
					candidates.push_back({value, 1});
				};
				auto majority = [&]() -> const Candidate * {
					for (const auto &candidate : candidates)
						if (candidate.votes > observations / 2)
							return &candidate;
					return nullptr;
				};
				Json checks = Json::array();
				auto record = [&](const LLMResponse &result) {
					checks.push_back({{"result_digest", ResultDigest(llm_detail::ParseJSON(result.raw_text).dump())},
					                  {"submit_ns", result.submit_ns},
					                  {"response_ns", result.response_ns},
					                  {"input_tokens", result.input_tokens},
					                  {"output_tokens", result.output_tokens}});
				};
				const auto initial = llm_detail::ParseJSON(provisional.raw_text);
				vote(initial);
				vote(latest);
				record(response);
				size_t retries = 0;
				while (!majority() && retries < max_correction_retries) {
					auto retry = request;
					retry.cache_correction_retry = true;
					// Each send passes the query ledger as a retransmission. Only
					// the initial check emits scheduler admission for this request.
					response = InferHTTP(retry, {});
					latest = llm_detail::ParseJSON(response.raw_text);
					vote(latest);
					record(response);
					++retries;
				}
				auto selected = majority();
				bool fallback = !selected;
				if (!selected) {
					size_t most = 0;
					std::vector<size_t> tied;
					for (size_t i = 0; i < candidates.size(); ++i) {
						if (candidates[i].votes > most) {
							most = candidates[i].votes;
							tied.clear();
						}
						if (candidates[i].votes == most)
							tied.push_back(i);
					}
					static thread_local std::mt19937_64 random(std::random_device {}());
					std::uniform_int_distribution<size_t> choose(0, tied.size() - 1);
					selected = &candidates[tied[choose(random)]];
				}
				winner = selected->value;
				response.cache_result_confirmed = true;
				response.cache_correction_fallback = fallback;
				response.cache_correction_calls = retries + 1;
				summary = {{"observations", observations},
				           {"winner_votes", selected->votes},
				           {"fallback", fallback},
				           {"correction_retries", retries}};
				response.cache_correction_metadata_json = Json({{"provisional_digest", ResultDigest(initial.dump())},
				                                                {"checks", checks},
				                                                {"summary", summary}})
				                                              .dump();
			}
			const Json envelope {{"kind", "semantic-result-cache"},
			                     {"version", 3},
			                     {"prompt_tokens", response.input_tokens},
			                     {"contract", contract},
			                     // Unchecked values remain provisional on disk so
			                     // enabling verification later still checks them.
			                     {"confirmed", verify_on_second_use && response.cache_result_confirmed},
			                     {"result", winner},
			                     {"verification", summary}};
			const auto proposed = envelope.dump();
			const auto published =
			    result_cache->Publish(request.result_cache_key, proposed, [&](const std::string &existing) {
				    try {
					    const auto candidate = llm_detail::ParseJSON(existing);
					    if (candidate.is_object() &&
					        candidate.value("kind", std::string()) == "semantic-result-cache" &&
					        candidate.at("version") == 3 && candidate.at("contract") == contract &&
					        candidate.at("confirmed").is_boolean() &&
					        (candidate.at("confirmed").get<bool>() || !verify_on_second_use)) {
						    ValidateStructuredPayload(request, candidate.at("result").dump());
						    return existing;
					    }
				    } catch (...) {
				    }
				    return proposed;
			    });
			if (published != proposed) {
				const auto authoritative = llm_detail::ParseJSON(published);
				winner = authoritative.at("result");
				ValidateStructuredPayload(request, winner.dump());
				response.cache_result_confirmed = authoritative.at("confirmed").get<bool>() || !verify_on_second_use;
				const auto verification = authoritative.value("verification", Json::object());
				response.cache_correction_fallback = verification.is_object() && verification.value("fallback", false);
				auto details = response.cache_correction_metadata_json.empty()
				                   ? Json::object()
				                   : llm_detail::ParseJSON(response.cache_correction_metadata_json);
				details["publication"] = {{"adopted_confirmed", response.cache_result_confirmed},
				                          {"result_digest", ResultDigest(winner.dump())}};
				response.cache_correction_metadata_json = details.dump();
			}
			const auto canonical = winner.dump();
			if (canonical != response.raw_text) {
				if (response.original_raw_text.empty())
					response.original_raw_text = response.raw_text;
				response.output_normalization =
				    winner == latest ? "semantic_cache_canonical_json" : "cache_correction_vote";
				response.raw_text = canonical;
			}
			finish();
			owner->set_value(response);
			return response;
		} catch (...) {
			// Failed verification leaves the original provisional entry intact.
			// In particular, budget exhaustion never invents a confirmation.
			auto error = std::current_exception();
			finish();
			owner->set_exception(error);
			std::rethrow_exception(error);
		}
	}
}

} // namespace semantic
