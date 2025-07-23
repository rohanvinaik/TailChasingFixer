#!/bin/bash

# Script to push the new improvements to GitHub

echo "🚀 Pushing Tail-Chasing Detector improvements to GitHub..."

# Navigate to the repository
cd /Users/rohanvinaik/github/TailChasingFixer

# Check git status
echo "📋 Current git status:"
git status

# Add all the new analyzer files
echo "📁 Adding new analyzer files..."
git add tailchasing/analyzers/tdd_antipatterns.py
git add tailchasing/analyzers/cross_file_duplication.py
git add tailchasing/analyzers/cargo_cult.py
git add tailchasing/analyzers/root_cause_tracer.py
git add tailchasing/analyzers/explainer.py

# Add the enhanced CLI
echo "📁 Adding enhanced CLI..."
git add tailchasing/cli_enhanced.py

# Add updated plugins.py
echo "📁 Adding updated plugins..."
git add tailchasing/plugins.py

# Add documentation
echo "📁 Adding documentation..."
git add docs/IMPLEMENTED_IMPROVEMENTS.md

# Add updated README
echo "📁 Adding updated README..."
git add README.md

# Show what will be committed
echo "📝 Files to be committed:"
git status --short

# Commit with a descriptive message
echo "💾 Committing changes..."
git commit -m "feat: Add advanced tail-chasing detection features

- Add TDD anti-pattern detector for mirror tests, brittle assertions
- Add cross-file semantic duplication analysis
- Add cargo cult programming pattern detection
- Add interactive root cause visualization with HTML reports
- Add natural language explanations for all patterns
- Add enhanced CLI with analyze, explain-pattern commands
- Update README with new features and examples

These enhancements significantly improve the tool's ability to detect
and explain LLM-induced anti-patterns, making it more valuable for
teams using AI-assisted development."

# Push to GitHub
echo "🌐 Pushing to GitHub..."
git push origin main

echo "✅ Successfully pushed improvements to GitHub!"
echo "🔗 View at: https://github.com/rohanvinaik/TailChasingFixer"
