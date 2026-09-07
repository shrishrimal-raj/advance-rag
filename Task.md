Research → Plan → Implement Pattern
keep in mind limit in the background-task limit (6 max running).
my main goal here is to learn RAG by doing practically as per industry standard so i request to help with me this in my goal by adding more examples for each module create multiple subagents or create an planning properly first everything should be perfect as per industry standard

i want you to create each and module with as detail as you can with multiple coding implementation approach multiple examples for each module as per industry practice, architecture diagram explanation from noob to subject expert professional goal update all docs or create new one as required as per planning in detail do deep research also i want to github portfolio add real world production use cases end to end with all docs code to prove my knowledge end to end from planning, designing, development, testing, production deployment, live to users

create cursor skill as well in this project folder to have proper guideline to follow read all files if needed solve this task using multiple sub agents for parallelly solving task

use this AI provider cloud openai compatible API documentation
Docs
OpenAI-compatible API documentation
Copy one setup prompt into your existing coding agent, or use the API reference and manual client configurations below.

Start here
Overview
Base URL, auth, models
Desktop app
API
OpenAI-compatible endpoints
Request format and vision
Capacity and retries
Request examples
Models page
Agent setup
One-prompt setup
Pi
OpenClaw
Hermes Agent
OpenCode
Aider
Cline
Roo Code
Continue
Claude Code bridge
Account
Get an API key
Pricing
Install with one prompt
Copy this prompt into the coding agent or AI editor you already use. It will identify the client, ask for your API key, add Yolo-Auto without removing existing providers, and verify the connection.

Copy install prompt
Configure the coding agent or AI code editor I am currently using to use Yolo-Auto as a model provider. Perform the setup for me instead of only describing it.

Provider details:
- Name: Yolo-Auto
- API protocol: OpenAI-compatible Chat Completions
- Base URL: https://yolo-auto.com/v1
- Default model: qwen3.8-27b
- Context window: 131072
- API key environment variable: YOLO_AUTO_API_KEY

Instructions:
- Identify the current coding agent or editor, its installed version, and its local user-level configuration before making changes.
- Ask me for my Yolo-Auto API key now if I have not provided it. Prefer a hidden credential prompt when available.
- Treat the API key as a secret. Never print the full key, place it in command history or logs, or store it in a repository.
- Use the agent's native custom OpenAI-compatible provider support when available. Preserve all existing providers and unrelated settings.
- Store the key in the agent's secure credential store or a user-level environment/config file that is excluded from version control.
- Add Yolo-Auto and make qwen3.8-27b available. Do not remove another provider or change my existing default model unless I ask.
- If this agent cannot use an OpenAI-compatible provider directly, use the smallest supported local compatibility bridge. Explain why it is needed and ask before installing a dependency.
- Validate the resulting configuration, then make an authenticated GET request to https://yolo-auto.com/v1/models or use the agent's safest equivalent connection check. Keep the key redacted.
- Finish by reporting the exact files or settings changed and the one command or action I should use to start coding with Yolo-Auto.
Base URL
https://yolo-auto.com/v1
Authentication
Authorization: Bearer yolo_...
Model IDs
"qwen3.8-27b"
Desktop app
Download the desktop app, grab other platform builds, or browse the source on GitHub.

Download for Windows
Download for Mac
Other releases
GitHub
Mac download is the Apple Silicon DMG. Intel Mac and Linux builds are available under Other releases.

Copy
Yolo-Auto Desktop v0.1.7
Windows download: https://github.com/yolo-auto-org/yolo-auto-desktop/releases/download/v0.1.7/YOLO-Auto-Desktop-Setup-0.1.7-x64.exe
Mac download: https://github.com/yolo-auto-org/yolo-auto-desktop/releases/download/v0.1.7/YOLO-Auto-Desktop-0.1.7-arm64.dmg
Other releases: https://github.com/yolo-auto-org/yolo-auto-desktop/releases
GitHub: https://github.com/yolo-auto-org/yolo-auto-desktop
HTTPS clone: git clone https://github.com/yolo-auto-org/yolo-auto-desktop
SSH clone: git clone git@github.com:yolo-auto-org/yolo-auto-desktop.git
OpenAI-Compatible API Endpoints
Use base URL https://yolo-auto.com/v1 and send Authorization: Bearer yolo_YOUR_KEY. Yolo-Auto implements a focused OpenAI-compatible surface rather than every OpenAI API route.

GET /v1/models
List the public model IDs available to compatible clients.
POST /v1/chat/completions
Send messages and create a normal or streaming chat completion.
GET /v1/usage
Read request, token, and quota metadata for the authenticated API key.
Use the request examples below to verify the endpoint, API key, model ID, JSON body, and streaming behavior before configuring a third-party client

draw mermaid diagram goal is to create multiple github portfolio projects to prove subject expert as per real world industry standard create awesome documentation as well markdown for everything as detail as you can please request

keep retrying until everything finishes coordinate all tasks carefully do proper planning and research before starting

Advanced RAG - Course Outline

Pre-requisite: LangGraph & LangChain Basics
Module 1: RAG Fundamentals and Architecture
Introduction to RAG (What it is and the problems it solves like hallucinations and knowledge cutoffs)
Core components and RAG data flow (Knowledge Base → Retriever → Generator)
Comparison: RAG vs Fine-Tuning vs Prompt Engineering
Module 2: Document Processing and Chunking
Document Loaders in LangChain (PDFs, Web, Structured data)
Text Splitting Strategies (Recursive, Character, Semantic, Markdown, and Code splitters)
Chunking Best Practices (Optimal chunk sizes and overlap strategies)
Metadata Management and filtering
Module 3: Embeddings and Vector Representations
How Embeddings Work (Vector representations, semantic similarity, and distance metrics)
Embedding Models Overview (OpenAI vs Open-source alternatives)
Choosing the Right Embedding Model (Cost vs quality trade-offs)
LangChain Embeddings Implementation
Module 4: Vector Stores
Vector Store Working and Indexing Strategies (IVF, HNSW)
Vector Store Operations (Create, Read, Update, Delete)
Module 5: Basic Retrieval Techniques
Similarity Search Fundamentals
Similarity Score Thresholds
Maximal Marginal Relevance (MMR) to balance relevance and diversity
Hybrid Search (Combining dense semantic vectors + sparse BM25 vectors)
Ensemble Retriever
Module 6: Advanced Retrieval Techniques
Contextual Compression
Parent Document Retriever
Self-Query 
Multi-Query Retrievers

Module 7: Advanced RAG Patterns
RAG Fusion and RRF
HyDE (Hypothetical Document Embeddings)
Corrective RAG (CRAG)
Self-RAG, 
Graph RAG
Multi Modal RAG
Module 8: Agentic RAG with LangGraph
Introduction to Agentic RAG and its capabilities
Using RAG as a Tool for Agents
LangGraph Fundamentals for RAG (State, nodes, edges, checkpointing)
Agentic RAG Design Patterns (ReAct, Plan-and-execute, Reflection)
Module 9: RAG evaluation through RAGAS
What is the RAGAS framework?
Parts of RAG pipeline to evaluate
RAG metrics in detail
Code implementation of individual metrics.
Code implementation of RAG pipeline using evaluate() API 
Module 10 : Capstone Project with Deployment (added very soon)
Project Assignment: Build a production-ready RAG system
Multi-document ingestion and hybrid retrieval with re-ranking
Agentic workflow with LangGraph
Evaluation suite (RAGAS) and Monitoring (LangSmith)
System Deployment
Production RAG
Pipeline Optimization
Caching Strategies
Cost Optimization
Monitoring and Debugging
Common Pitfalls and Solutions
Security and Compliance


Create an agent skill for this Goose AI Desktop
Skip to main content
✨ goose has moved to the Agentic AI Foundation (AAIF): Learn more! ✨

goose Logo
Quickstart
Docs
Tutorials
Blog
Resources
Discord
GitHub


Quickstart
Getting Started

GDK

Guides

Managing Sessions

Context Engineering

Using goosehints
Creating Plans
Subagents
Custom Agents
Agent Skills
Custom Slash Commands
Hooks
Plugins
Prompt Templates
Persistent Instructions
Memory Extension
Research → Plan → Implement
Recipes

Managing Tools

Updating goose
CLI Commands
CLI Providers
ACP Providers
Configuration Files
Environment Variables
Quick Tips
Tool Shim
Security

MCP Sampling
MCP Apps

MCP Elicitation
MCP Roots
Custom Distributions
LLM Rate Limits
Logging System
Usage Data
File Management
Run Tasks
Extension Allowlist
Remote Server
Offline Docs
Roaming Agents
Multi-Model Config
Codebase Analysis
Azure AI Foundry
Customizing the Sidebar
VMware Tanzu Platform
VMware Tanzu Platform - CLI Testing Guide
Terminal Integration
Tutorials

MCP Servers

Architecture Overview

Experimental

Troubleshooting

GuidesContext EngineeringAgent Skills
Agent Skills
Copy page

Skills are reusable sets of instructions and resources that teach goose how to perform specific tasks. A skill can range from a simple checklist to a detailed workflow with domain expertise, and can include supporting files like scripts or templates. Example use cases include deployment procedures, code review checklists, and API integration guides.

info
This functionality uses the built-in Skills platform extension, which is enabled by default.

When a session starts, goose adds discovered skill names and descriptions to its instructions. During the session, goose can load a skill's full instructions when:

Your request clearly matches a skill's purpose
You explicitly ask to use a skill, for example:
"Use the code-review skill to review this PR"
"Follow the new-service skill to set up the auth service"
"Apply the deployment skill"
You can also ask goose what skills are available, run goose skills list, or use the CLI /skills command to list available skills and load one or more by name:

/skills code-review edge-case-finder

Claude Compatibility
goose skills are compatible with Claude Desktop and other agents that support Agent Skills.

Built-in Skills
goose ships with a built-in skill that is always available without any installation:

Skill	Description
web-search	Search the web using DuckDuckGo (no API key), Tavily, or SearXNG, and extract page content.
For browser automation — navigating pages, clicking, filling forms, and capturing screenshots — install the upstream-maintained browser-use skill:

browser-use skill install

This gives you the full, up-to-date skill from the browser-use project, including remote browser support, the AX-tree element selection strategy, and recording tools.

Skill Locations
Skills can be stored globally, per-project, or in installed plugins:

~/.agents/skills/ — Global skills, available in all sessions
.agents/skills/ — Project-level skills, scoped to the current project
~/.agents/plugins/<plugin-name>/ — Skills provided by installed plugins
Place a SKILL.md file inside a named subdirectory. For example, a global skill called code-review goes in ~/.agents/skills/code-review/SKILL.md.

Backward compatibility: goose also discovers skills from .goose/skills/, .claude/skills/, ~/.claude/skills/, and platform-specific config directories, but agents/skills/ is the recommended standard.

Creating a Skill
Create a skill when you have a repeatable workflow that involves multiple steps, specialized knowledge, or supporting files.

Skill File Structure
Each skill lives in its own directory with a SKILL.md file:

~/.agents/skills/
└── code-review/
    └── SKILL.md

A SKILL.md file requires YAML frontmatter with name and description, followed by the skill content:

---
name: code-review
description: Comprehensive code review checklist for pull requests
---

# Code Review Checklist

When reviewing code, check each of these areas:

## Functionality
- [ ] Code does what the PR description claims
- [ ] Edge cases are handled
- [ ] Error handling is appropriate

## Code Quality
- [ ] Follows project style guide
- [ ] No hardcoded values that should be configurable
- [ ] Functions are focused and well-named

## Testing
- [ ] New functionality has tests
- [ ] Tests are meaningful, not just for coverage
- [ ] Existing tests still pass

## Security
- [ ] No credentials or secrets in code
- [ ] User input is validated
- [ ] SQL queries are parameterized

Skills from Plugins
Skills can also come from installed plugins. Plugin-provided skills are discovered at session startup and work like other skills. For Open Plugins, skill names are namespaced with the plugin name, such as my-plugin:review. Use that full name when explicitly loading a plugin-provided skill.

Supporting Files
Skills can include supporting files like scripts, templates, or configuration files. Place them in the skill directory:

~/.agents/skills/
└── api-setup/
    ├── SKILL.md
    ├── setup.sh
    └── templates/
        └── config.template.json

When goose loads the skill, it sees the supporting files and can access them using the Developer extension's file tools.

Example Skill with Supporting Files
Common Use Case Examples
Deployment Workflow
Testing Strategy
API Integration Guide
Other goose features that support reuse
.goosehints: Best for general preferences, project context, and repeated instructions like "Always use TypeScript"
recipes: Shareable configurations that package instructions, prompts, and settings together
Best Practices
Keep skills focused — One skill per workflow or domain. If a skill is getting long, consider splitting it.
Write for clarity — Skills are instructions for goose. Use clear, direct language and numbered steps.
Include verification steps — Help goose confirm the workflow completed successfully.
Additional Resources
Thumbnail for Did Skills Kill MCP?
Previous
Custom Agents
Next
Custom Slash Commands
Built-in Skills
Skill Locations
Creating a Skill
Skill File Structure
Skills from Plugins
Supporting Files
Common Use Case Examples
Best Practices
Additional Resources
Quick Links
Install goose
Extensions
Community
Spotlight
Discord
YouTube
LinkedIn
Twitter / X
BlueSky
Nostr
More
Blog
GitHub
Copyright © 2026 AAIF (Agentic AI Foundation)

do web search for if you want more context:
https://goose-docs.ai/docs/guides/context-engineering/using-skills/

I've grasped the overall picture. Now I'll execute the Research → Plan → Implement pattern:

Research findings: All 11 modules are complete and verified (v1). What's needed for the v2 expansion: multi-approach code per module, beginner→expert diagrams, 3 GitHub portfolio projects, Cursor + Goose skills, Yolo-Auto integration, and detailed documentation.

Step 1 — Plan: Create MASTER_PLAN.md and set up the infrastructure (Yolo-Auto provider, skills, dependencies).

AGENTS.md                                                                                                                            
Subagent Delegation Limits (provider constraint)
Provider cap: ~128K context per coding agent, typically 3-4 agents concurrent.                                                       
One focused task per agent; give narrow file scope + verified line ranges in the brief (pre-map big files yourself, do not make the  
agent rediscover structure).                                                                                                         
Never hand a single agent a job that requires reading many >2000-line files end-to-end; split across parallel background agents by   
file/module.                                                                                                                         
Briefs must be self-contained (agent has no shared context): exact paths, function names, acceptance checks, "no commit/push" when   
applicable.                                                                                                                          
Expect quality drop after ~60-70% context on long tasks; prefer several small agents over one marathon agent.                        
Main session always verifies agent output (lint/build/behavioral probe) before reporting done


solve this task carefully as per instruction given Task.md create a checkpoint file to track
update check point as well to track whats completed whats left carefully  