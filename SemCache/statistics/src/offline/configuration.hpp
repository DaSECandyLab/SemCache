#pragma once

#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

namespace semcache {
inline nlohmann::json ReadConfiguration(const std::filesystem::path &path) {
  auto read = [](const std::filesystem::path &source) {
    std::ifstream stream(source);
    if (!stream)
      throw std::runtime_error("Cannot read configuration: " + source.string());
    auto value = nlohmann::json::parse(stream, nullptr, true, true);
    if (!value.is_object())
      throw std::invalid_argument("Configuration must be a JSON object: " +
                                  source.string());
    return value;
  };
  auto settings = read(path);
  if (!settings.contains("include"))
    return settings;
  const auto &modules = settings.at("include");
  if (settings.size() != 1 || !modules.is_array() || modules.empty())
    throw std::invalid_argument(
        "Configuration index requires only a nonempty include list");
  const auto source = std::filesystem::weakly_canonical(path);
  nlohmann::json merged = nlohmann::json::object();
  std::set<std::filesystem::path> sources;
  for (const auto &name : modules) {
    if (!name.is_string() || name.get<std::string>().find_first_not_of(
                                 " \t\r\n") == std::string::npos)
      throw std::invalid_argument(
          "Configuration module path must be a nonempty string");
    auto module_path = std::filesystem::weakly_canonical(
        source.parent_path() / name.get<std::string>());
    if (module_path == source || !sources.insert(module_path).second)
      throw std::invalid_argument("Repeated configuration module: " +
                                  module_path.string());
    const auto module = read(module_path);
    if (module.empty() || module.contains("include"))
      throw std::invalid_argument(
          "Configuration module must contain sections: " +
          module_path.string());
    for (const auto &section : module.items()) {
      if (merged.contains(section.key()))
        throw std::invalid_argument("Duplicate configuration section: " +
                                    section.key());
      merged[section.key()] = section.value();
    }
  }
  return merged;
}
}
