//
// Created by creeper on 10/24/24.
//
#include <string>
#include <iostream>
#include <Core/reflection.h>
using namespace ksk::core;

struct Student {
  std::string name;
  int age;
  double gpa;
};
REFLECT(Student, name, age, gpa)

int main() {
  Student s{"John Doe", 20, 3.5};
  reflect<Student>::forEachMember(s, [&](const char * memberName, auto &member) {
    std::cout << memberName << ": " << member << std::endl;
  });
  return 0;
}