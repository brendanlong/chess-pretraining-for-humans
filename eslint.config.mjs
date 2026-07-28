import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["web/vendor/**"] },
  js.configs.recommended,
  {
    files: ["web/**/*.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: globals.browser,
    },
  },
];
