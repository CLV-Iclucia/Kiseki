//
// Created by CreeperIclucia-Vader on 25-5-26.
//
#include <Core/json.h>
#include <charconv>
#include <fstream>
#include <regex>

namespace ksk::core {

template <typename T> std::optional<T> tryParseNumber(std::string_view str) {
  T value;
  if (auto result = std::from_chars(str.data(), str.data() + str.size(), value);
      result.ec == std::errc() && result.ptr == str.data() + str.size())
    return value;
  return {};
}

static char unescape(char ch) {
  switch (ch) {
  case 'b':
    return '\b';
  case 'f':
    return '\f';
  case 'n':
    return '\n';
  case 'r':
    return '\r';
  case 't':
    return '\t';
  case 'v':
    return '\v';
  default:
    return ch;
  }
}

struct JsonParseResult {
  JsonNode node;
  size_t consumed;
};

static JsonParseResult parseJsonImpl(std::string_view json) noexcept {
  if (json.empty())
    return {JsonNode{std::nullptr_t()}, 0};

  if (size_t offset = json.find_first_not_of(" \t\n\r\v\f\0");
      offset && offset != std::string::npos) {
    auto [node, consumed] = parseJsonImpl(json.substr(offset));
    return {node, offset + consumed};
  }

  char lookAhead = *json.begin();
  if (std::isdigit(lookAhead) || lookAhead == '+' || lookAhead == '-') {
    std::regex numberRegex(R"([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)");
    if (std::cmatch match; std::regex_search(
            json.data(), json.data() + json.size(), match, numberRegex)) {
      std::string str = match.str();
      if (auto value = tryParseNumber<int>(str))
        return {JsonNode{*value}, str.size()};
      if (auto value = tryParseNumber<Real>(str))
        return {JsonNode{*value}, str.size()};
    }
    return {JsonNode{std::nullptr_t()}, 0};
  }
  if (lookAhead == 't') {
    if (json.size() < 4 || json.substr(0, 4) != "true")
      return {JsonNode{std::nullptr_t{}}, 0};
    return {JsonNode{true}, 4};
  }
  if (lookAhead == 'f') {
    if (json.size() < 5 || json.substr(0, 5) != "false")
      return {JsonNode{std::nullptr_t{}}, 0};
    return {JsonNode{false}, 5};
  }
  if (lookAhead == 'n') {
    if (json.size() < 4 || json.substr(0, 4) != "null")
      return {JsonNode{std::nullptr_t{}}, 0};
    return {JsonNode{std::nullptr_t{}}, 4};
  }

  if (lookAhead == '"') {
    std::string str;
    enum {
      Raw,
      Escaped,
    } phase = Raw;
    size_t i = 1;
    for (; i < json.size(); i++) {
      char ch = json[i];
      if (phase == Raw) {
        if (ch == '\\')
          phase = Escaped;
        else if (ch == '"') {
          i++;
          break;
        } else
          str += ch;
      } else {
        str += unescape(ch);
        phase = Raw;
      }
    }
    return {JsonNode{str}, i};
  }

  auto skipWhitespaces = [&](size_t &i) {
    if (auto offset = json.substr(i).find_first_not_of(" \t\n\r\v\f\0");
        offset && offset != std::string::npos)
      i += offset;
  };
  if (lookAhead == '[') {
    JsonList list{};
    size_t i = 1;
    bool enclosed = false;
    bool expectComma = false;

    while (i < json.size()) {
      skipWhitespaces(i);

      if (json[i] == ']') {
        i++;
        enclosed = true;
        break;
      }

      if (expectComma) {
        if (json[i] != ',') {
          i = 0;
          break;
        }
        i++;
        skipWhitespaces(i);
      }

      auto [node, consumed] = parseJsonImpl(json.substr(i));
      if (!consumed) {
        i = 0;
        break;
      }
      list.push_back(std::move(node));
      i += consumed;
      expectComma = true;
    }

    if (enclosed)
      return {JsonNode{list}, i};
  }

  if (lookAhead == '{') {
    std::unordered_map<std::string, JsonNode> dict{};
    size_t i = 1;
    bool enclosed = false;
    bool expectComma = false;

    while (i < json.size()) {
      skipWhitespaces(i);

      if (json[i] == '}') {
        i++;
        enclosed = true;
        break;
      }

      if (expectComma) {
        if (json[i] != ',') {
          i = 0;
          break;
        }
        i++;
        skipWhitespaces(i);
      }

      auto [key, consumed] = parseJsonImpl(json.substr(i));
      if (!key.is<std::string>()) {
        i = 0;
        break;
      }
      if (!consumed) {
        i = 0;
        break;
      }
      i += consumed;
      skipWhitespaces(i);

      if (json[i] == ':')
        i++;
      else {
        i = 0;
        break;
      }

      auto [value, consumedValue] = parseJsonImpl(json.substr(i));
      if (!consumedValue) {
        i = 0;
        break;
      }
      i += consumedValue;
      dict.insert_or_assign(key.as<std::string>(), std::move(value));
      expectComma = true;
    }

    if (enclosed)
      return {JsonNode{dict}, i};
  }
  return {JsonNode{std::nullptr_t{}}, 0};
}

std::optional<JsonNode> parseJson(std::string_view json) noexcept {
  if (auto [node, consumed] = parseJsonImpl(json); consumed == json.size())
    return node;
  return {};
}

std::optional<JsonNode>
loadJsonFile(const std::filesystem::path &path) noexcept {
  auto file = std::ifstream(path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (auto node = parseJson(content))
    return node;
  return std::nullopt;
}

} // namespace core