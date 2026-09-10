#!/usr/bin/env node

import { main } from "../src/cli.js";

main(process.argv.slice(2)).catch((error) => {
  console.error(`codehelm: ${error.message}`);
  if (process.env.CODEHELM_DEBUG) console.error(error.stack);
  process.exitCode = 1;
});
