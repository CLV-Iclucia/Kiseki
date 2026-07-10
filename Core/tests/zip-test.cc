//
// Created by creeper on 6/3/24.
//
#include <Core/zip.h>
#include <gtest/gtest.h>

using namespace ksk::core;
TEST(ZipTest, ZipTwoVectors) {
  std::vector<int> a = {1, 2, 3, 4};
  std::vector<int> b = {4, 5, 6, 7};
  for (auto [x, y] : zip(a, b)) {
    EXPECT_EQ(x + 3, y);
  }
}

TEST(ZipTest, ZipModify) {
  std::vector<int> a = {1, 2, 3, 4};
  std::vector<int> b = {4, 5, 6, 7};
  for (auto [x, y] : zip(a, b)) {
    x = 0;
  }
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], 0);
  }
}

TEST(ZipTest, ZipThreeVectors) {
  std::vector<int> a = {1, 2, 3, 4};
  std::vector<int> b = {4, 5, 6, 7};
  std::vector<int> c = {5, 7, 9, 11};
  for (auto [x, y, z] : zip(a, b, c)) {
    EXPECT_EQ(x + y, z);
  }
}

TEST(ZipTest, ZipVectorOfVectors) {
    std::vector<std::vector<int>> a = {{1, 2, 3, 4},
                                        {5, 6, 7, 8},
                                        {9, 10, 11, 12}};
    std::vector<std::vector<int>> b = {{4, 5, 6, 7},
                                        {8, 9, 10, 11},
                                        {12, 13, 14, 15}};
    for (auto [x, y] : zip(a, b)) {
        for (auto [i, j] : zip(x, y)) {
            EXPECT_EQ(i + 3, j);
        }
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}