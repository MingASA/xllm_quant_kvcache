include(cc_test)

cc_test(
  NAME
    native_fp8_config_test
  CPU_ONLY
  SRCS
    native_fp8_config_test.cpp
  DEPS
    :config
    :util
    GTest::gtest_main
)

cc_test(
  NAME
    config_json_test
  CPU_ONLY
  SRCS
    config_json_test.cpp
  DATA
    config_test.json
  DEPS
    :config
    :util
    GTest::gtest_main
)
