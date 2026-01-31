---
agent: agent
description: 'Create a new instructions file from the template'
---

## Task
create an instructions file from the template

Your task is to create an instructions file from the template to serve this repository to a coding agent for Language or framework-specific rules. It should contain information describing how the agent, seeing the repo for the first time, can work most efficiently.

You will do this task only one time per repository for the specific instruction, and doing a good job can SIGNIFICANTLY improve the quality of the agent's work, so take your time, think carefully, and search thoroughly before writing the instructions

When you receive a request, create a new instructions file based on
<template>
---
applyTo: <blob-path>
---

# Title
description

# Goal
description

## Essentials
- list item(s)

## Tech Stack
- list item(s) as markdown links, (ex. `[tech](https://www.tech.org/)`)

##  Project Structure
markdown tree diagram

## Key files
- list item(s)

## Development Guidelines
### Other Headings
description
examples

## Reference Resources
- list item(s)
</template>


## Expectations
- Require a single `name` argument (base name only, no path or extension).
- If `name` is not provided, ask: "Please provide a name for the instructions file (base name only):" and wait for user input.
- The created file path must be `.github/instructions/<name>.instructions.md` (note the exact spelling of the extension).
- Do not overwrite an existing file unless the user explicitly confirms. If the target file exists, ask: "File `.github/instructions/<name>.instructions.md` exists. Confirm overwrite? (yes/no)" and wait for a clear affirmative `yes` before overwriting.

Interaction examples:
- User: `create instructions name=onboarding` -> Create `.github/instructions/onboarding.instructions.md` from template, filling `{{name}}` with `onboarding`.
- User: `create instructions` -> Ask for `name` before proceeding.

Safety:
- Never create files outside `.github/instructions/` unless the user explicitly requests a different directory.

Notes:
- Ensure the file is valid Markdown and follows the structure of the template.
- Use relevant project details to fill sections like Tech Stack, Key Files, etc.
- If unsure about project specifics, use placeholders and ask the user for clarification.
- Ensure the generated file is optimized for clarity, usefulness to developers, and token efficiency.
- After creating the file, respond with: "Instructions file `.github/instructions/<name>.instructions.md` created successfully."

