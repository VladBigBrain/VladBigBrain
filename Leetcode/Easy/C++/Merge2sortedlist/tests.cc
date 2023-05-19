#include <gtest/gtest.h>
TEST(AddTest, PositiveNumbers) { EXPECT_EQ(2 + 3, 5); }
TEST(AddTest, NegativeNumbers) { EXPECT_EQ(-2 - 3, -5); }
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
