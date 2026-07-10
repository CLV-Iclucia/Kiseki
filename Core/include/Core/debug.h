//
// Created by creeper on 24-3-20.
//

#pragma once
#include <iostream>
#include <functional>
#include <type_traits>
#include <format>
#include <source_location>
#include <Core/properties.h>
namespace ksk::core {
inline void do_nothing() {};
// only support single thread for now
class Debugger {
 public:
  void activate() {
    activated = true;
  }
  void deactivate() {
    activated = false;
  }
  template<typename Func>
  void breakpoint(Func &&func, std::source_location loc = std::source_location::current()) {
    if (activated) {
      std::cout << std::format("Triggered breakpoint at {}:{} in function {} by debugger {}\n",
                               loc.file_name(),
                               loc.line(),
                               loc.function_name(),
                               debugger_name);
      func();
      printf("Breakpoint callback finished, launch debugging console\n");
      debugConsole();
    }
  }
  void breakpoint(std::source_location loc = std::source_location::current()) {
    if (activated) {
      std::cout << std::format("Triggered breakpoint at {}:{} in function {} by debugger {}\n",
                               loc.file_name(),
                               loc.line(),
                               loc.function_name(),
                               debugger_name);
      breakpoint_callback();
      printf("Breakpoint callback finished, launch debugging console\n");
      debugConsole();
    }
  }
  explicit Debugger(std::string_view name) : activated(false), debugger_name(name), breakpoint_callback(do_nothing) {}

 protected:
  bool activated{false};
  std::string debugger_name{"Default Debugger"};
  std::function<void(void)> breakpoint_callback{do_nothing};
  void debugConsole() {
    while (true) {
      std::cout << std::format("[DEBUG CONSOLE | {}] ", debugger_name);
      std::string command;
      std::getline(std::cin, command);
      if (command == "deactivate") {
        deactivate();
        return;
      } else if (command == "exit") {
        std::cout << "Exiting debug console\n";
        return;
      } else
        std::cout << "Unknown command" << std::endl;
    }
  }
  template<typename Func>
  void registerBreakpointCallback(Func &&callback) {
    breakpoint_callback = callback;
  }
};
}
