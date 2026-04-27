✦ To add the changelog-manager skill, use the following commands based on your preference:

  1. Locally (Workspace Scope)
  Use this if you only want the skill available within this specific project:

   1 gemini skills install changelog-manager.skill --scope workspace

  2. User Level (User Scope)
  Use this to make the skill available across all your projects:

   1 gemini skills install changelog-manager.skill --scope user

  3. Enable the Skill
  After running the installation command, you must manually execute the reload command in your
  interactive Gemini CLI session:
   1 /skills reload

  You can then verify it is active by running /skills list.