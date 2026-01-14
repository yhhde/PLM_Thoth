# 🤝 PLM_Thoth GitHub Collaboration Guide

> **Team collaboration workflow for PLM_Thoth course project**

---

## 🚀 Quick Start

This guide covers **Git setup** and **daily collaboration workflow**.

**Contents:**
- Getting started
- Daily Git workflow (pull, commit, push)
- Conflict resolution
- Best practices

---

## 👥 Team Information

**Project:** PLM_Thoth - Bilingual GPT-2 Pretraining  
**Repository:** `PLM_Thoth` (Private 🔒)  
**URL:** `https://github.com/yhhde/PLM_Thoth`

**Team Members:**
- **hyonghua** (Admin) 
- **wstempniak** (Write) 
- **anwaecht** (Write) 

**📝 Path Convention:**
```
GitHub Repository:  PLM_Thoth
                          ↓
Code Path:          /nethome/$USER/PLM_Thoth/
Data Path:          /scratch/$USER/thoth_project/
```

---

## 📁 Project Structure

```
PLM_Thoth/
├── 📄 Core Scripts
│   ├── 0_bootstrap_tokenizer.py    # Tokenizer initialization
│   ├── 1_download_dataset.py       # Data download
│   ├── 2_preprocess_and_split.py   # Data preprocessing
│   ├── 2+_mono_and_bucket.py       # Mono and Bucketing processing
│   ├── 3_train_tokenizer.py        # Tokenizer training
│   ├── 4_pretokenize.py            # Pre-tokenization
│   ├── 5_train_model.py            # Model training ⭐
│   ├── 6_validation.py             # Model validation ⭐
│   └── create_medium_dataset.py    # Create medium dataset
│
├── 📄 Model & Experiments
│   ├── model.py                    # GPT-2 model definition
│   ├── run_experiments.py          # Batch experiment runner
│   ├── experiments_*.jsonl         # Experiment configs
│   ├── analyze_experiments.py      # Experiment analysis
│   └── visualize_experiments.py    # Results visualization
│
├── 📄 Config & Utils
│   ├── requirements.txt            # Python dependencies
│   ├── stats.json                  # Data statistics
│   └── test_cuda.py                # CUDA test
│
└── 📁 docs/                        # Documentation
    └── GITHUB_COLLABORATION.md     # This document
```

---

## 📥 Getting Started

### Step 1: Navigate to Repository

```bash
# SSH to cluster
ssh your_username@cluster

# Navigate to repository
cd /nethome/$USER/PLM_Thoth

# Configure Git (if not done)
git config user.name "Your Name"
git config user.email "your@email.com"
```

### Step 2: Environment Setup

```bash
# Activate conda environment
conda activate thoth

# Install dependencies (if needed)
pip install -r requirements.txt

# Login to WandB
wandb login
```

---

## 🔄 Daily Workflow

### Before Making Changes

**Always pull first!**

```bash
cd /nethome/$USER/PLM_Thoth
git pull origin main
```

### Making Changes

```bash
# 1. Check changed files
git status

# 2. Stage your changes
git add <file>
# Or stage all:
git add .

# 3. Commit with descriptive message
git commit -m "Fix: Correct tokenizer padding issue"

# 4. Push to GitHub
git push origin main
```

---

## 📝 Commit Message Guidelines

### Format

```
Type: Brief description
```

### Types

- `Fix:` - Bug fixes
- `Add:` - New features
- `Update:` - Modify existing code
- `Docs:` - Documentation changes
- `Refactor:` - Code cleanup
- `Config:` - Configuration changes

### Examples

✅ **Good:**
```bash
git commit -m "Fix: Correct next-token shift in training loop"
git commit -m "Add: Conditional probability retrieval method"
git commit -m "Update: Add logging to 5_train_model.py"
```

❌ **Bad:**
```bash
git commit -m "fix bug"        # Too vague
git commit -m "update"         # What was updated?
```

---

## ⚠️ Conflict Resolution

### When Conflicts Occur

```bash
$ git push origin main
! [rejected] main -> main (fetch first)
error: failed to push
```

### Solution

```bash
# 1. Pull changes
git pull origin main

# 2a. If auto-merge succeeds:
git push origin main  # Done!

# 2b. If conflict occurs:
# Open conflicted files, look for markers:
<<<<<<< HEAD
your changes
=======
their changes
>>>>>>> main

# Edit to resolve, remove markers, save file

# 3. Stage resolved files
git add <resolved-file>

# 4. Complete merge
git commit -m "Merge: Resolved conflict in training config"

# 5. Push
git push origin main
```

---

## ✅ Best Practices

### DO ✅

1. **Pull before starting work**
2. **Commit frequently** (small, logical changes)
3. **Write descriptive commit messages**
4. **Push regularly**
5. **Test before pushing**

### DON'T ❌

1. **Don't commit large files**
   - Data files (datasets/)
   - Model files (checkpoints/)
   - WandB logs (wandb/)

2. **Don't force push**
   ```bash
   git push --force  # NEVER! Loses teammates' work
   ```

3. **Don't leave uncommitted changes too long**

4. **Don't push broken code**

---

## 🌿 Branch Strategy

### Small Changes: Use main

```bash
git pull origin main
# Make changes
git commit -m "Fix: Bug description"
git push origin main
```

### Major Changes: Use Feature Branch

```bash
# Create feature branch
git checkout -b feature/new-validation

# Work and commit
git commit -m "Add: New validation method"

# Push feature branch
git push -u origin feature/new-validation

# After testing, merge to main
git checkout main
git merge feature/new-validation
git push origin main

# Delete feature branch
git branch -d feature/new-validation
```

---

## 📋 Quick Reference

### Essential Commands

```bash
# Daily workflow
git pull origin main              # Pull latest
git status                        # Check changes
git add .                         # Stage all
git commit -m "Description"       # Commit
git push origin main              # Push

# View history
git log --oneline                 # Recent commits
git diff                          # See changes

# Undo changes
git checkout -- <file>            # Discard changes
git reset HEAD <file>             # Unstage file

# Branches
git checkout -b branch-name       # Create branch
git checkout main                 # Switch to main
```

---

## 🔗 Important Files

| File | Purpose | Notes |
|------|---------|-------|
| `5_train_model.py` | Model training | Fixed next-token prediction shift |
| `6_validation.py` | Model validation | Uses conditional probability method |
| `model.py` | GPT-2 model | Core architecture definition |
| `experiments_*.jsonl` | Experiment configs | Hyperparameter settings |

---

## 💡 Collaboration Tips

### Avoiding Conflicts

1. **Work on different files** when possible
2. **Communicate** before editing same file
3. **Pull frequently** to stay in sync
4. **Push completed work promptly**

### Efficient Workflow

1. **Small, frequent commits** > Large, infrequent
2. **Test locally** before pushing
3. **Document as you go**
4. **Review teammates' commits** (learn from each other)

---

## 📞 Getting Help

**Git Help:**
- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com

**Project Help:**
- Create GitHub Issue for bugs
- Discuss on WandB team page

---

**Happy Collaborating! 🚀**
