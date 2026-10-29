import Lake
open Lake DSL

package "LeanSandbox" where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.15.0"

@[default_target]
lean_lib «LeanSandbox» where
