#!/bin/bash
# Script to commit and push the advanced features

cd /Users/rohanvinaik/github/TailChasingFixer

# Check git status
echo "📊 Git Status:"
git status --short

echo -e "\n📝 Adding new files..."
# Add all the new advanced feature files
git add tailchasing/analyzers/advanced/
git add tailchasing/fixers/advanced/
git add examples/advanced_demo.py
git add docs/ADVANCED_FEATURES.md

# Add modified files
git add tailchasing/plugins.py
git add tailchasing/core/scoring.py
git add README.md

echo -e "\n💾 Creating commit..."
git commit -m "feat: Add advanced tail-chasing detection features

- Add Hallucination Cascade Detection for fictional subsystems
- Add Context Window Thrashing Detection for lost context reimplementations  
- Add Import Anxiety Pattern Detection for defensive over-importing
- Add Enhanced Semantic Analysis with multi-modal encoding
- Add Intelligent Auto-Fix System with confidence scores and impact analysis
- Update plugins system to support advanced analyzers
- Add comprehensive documentation and demo script
- Update README with new advanced features

Based on real-world testing feedback and improvements suggestions"

echo -e "\n🚀 Pushing to GitHub..."
git push origin main

echo -e "\n✅ Done! Advanced features pushed to GitHub."
