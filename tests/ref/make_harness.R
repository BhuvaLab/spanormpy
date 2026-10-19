source("tests/ref/_common.R")
write_ref(list(
  m = matrix(1:6 + 0.5, 2, 3),
  v = c(1, NA),
  s = c("a", "b"),
  x = 3.25,
  f = factor(c("u", "v", "u"))
), "harness")
