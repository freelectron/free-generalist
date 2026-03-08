# Free Generalist
A simple agent that only uses local and free compute/llms.

## Current Functionality

### Search the Web 

Agent searches the web and finds answers to your questions. 

### Writer some Code

Agent can write, execute, analyse code. 

### Read Files

Agent can read files that are in text format, i.e., can load text from files that have binaries which are ASCII or UTF-8
encoded characters.  


## To Be Implemented 
 
### Let the app be helpful with coding

You point the agent to a file, tell it what it needs to do, e.g., code refactoring, create a class with 
specific functionality. It looks at the code of that file, sees if it needs additional context, e.g., 
look up other files and keep them in the scope. Implements the code changes by creating a copy of the file, 
i.e., no LSP (Language Server Protocol) integration, just plain copying.

Can look into `opencode` for some nice [docs](https://deepwiki.com/sst/opencode/5-tool-system) of tool examples. 

### Live Interaction with Websites

*Option 1*
Make use of something like [`browser-use`](https://github.com/browser-use/browser-use?tab=readme-ov-file) 
to interact with web pages. It is especially interesting to see how `browser-use` uses `playwrite` (which uses Chrome Dev Protocol)
to parse any html/js page and extract actionable/relevant page objects (buttons, text, etc) from there.
See deepwiki for some more context on [DOM parsing](https://deepwiki.com/browser-use/browser-use/5-dom-processing-engine) in `browser-use`.

*Option 2*
Use `selenium` to do page parsing and doing an action. Gives a lot of freedom on how to do things.  