# Agent Operating Constraints

## Software Installation

- Do not install, upgrade, or remove software on the local machine, remote
  servers, or development boards unless the user gives explicit approval for
  that exact action.
- This restriction includes package managers and installers such as `apt`,
  `pip`, `conda`, `npm`, `yarn`, `pnpm`, `brew`, system package tools, and
  vendor SDK installers.
- It is acceptable to inspect existing environments, check versions, and test
  imports without changing installed software.
- If a missing dependency blocks a task, report the missing dependency and
  provide the exact command for the user to run, instead of running it
  automatically.

## Python Runtime

- The local machine, remote servers, and Ascend 310B development boards already
  have Anaconda installed. Run project Python programs only inside the relevant
  Anaconda/conda environment.
- Do not run project Python programs with system Python unless the user
  explicitly requests it for a specific diagnostic.
- Do not run `conda init`, edit shell startup files such as `.bashrc`, `.zshrc`,
  or `.profile`, or otherwise change login-shell initialization unless the user
  explicitly approves that exact change.
- For non-interactive SSH commands, activate conda only within that command
  session, for example by sourcing the existing conda profile script and then
  running `conda activate <env>`.
- If a conda environment cannot be activated, stop and report the activation
  failure instead of falling back to system Python or installing packages.
