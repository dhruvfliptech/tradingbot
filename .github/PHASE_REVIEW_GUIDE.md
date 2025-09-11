# GitHub Phase Review Guide - AI Trading Bot

## Phase-Based Development Structure

This guide provides GitHub configuration and workflows for phase-by-phase CTO review with AI agent compatibility.

---

## Branch Protection Rules

### Main Branch Protection
```yaml
# Settings > Branches > Add rule
Pattern: main
Required reviews: 2
Dismiss stale reviews: true
Require branches up to date: true
Require conversation resolution: true
Require signed commits: false
Include administrators: false
```

### Phase Branch Protection
```yaml
# For each phase-X branch
Pattern: phase-*
Required reviews: 1
Require status checks: true
- ci/build
- ci/test
- ci/lint
```

---

## GitHub Actions Workflows

### 1. Phase Validation Workflow
Create `.github/workflows/phase-validation.yml`:

```yaml
name: Phase Validation

on:
  pull_request:
    branches: [ main, phase-* ]
  push:
    branches: [ phase-* ]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run TypeScript check
      run: npx tsc --noEmit
    
    - name: Run linting
      run: npm run lint
    
    - name: Build project
      run: npm run build
    
    - name: Check bundle size
      run: |
        npm run build
        size=$(du -sb dist | cut -f1)
        echo "Bundle size: $size bytes"
        if [ $size -gt 5000000 ]; then
          echo "Bundle too large!"
          exit 1
        fi
    
    - name: Run tests (if available)
      run: npm test --if-present
    
    - name: Generate phase report
      run: |
        echo "# Phase Validation Report" > phase-report.md
        echo "## Build Status: ✅" >> phase-report.md
        echo "## TypeScript: ✅" >> phase-report.md
        echo "## Linting: ✅" >> phase-report.md
        echo "## Bundle Size: $size bytes" >> phase-report.md
        
    - name: Upload phase report
      uses: actions/upload-artifact@v3
      with:
        name: phase-report
        path: phase-report.md
```

### 2. Performance Benchmark Workflow
Create `.github/workflows/performance-benchmark.yml`:

```yaml
name: Performance Benchmark

on:
  pull_request:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'backend/**'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup environment
      run: |
        npm ci
        cd backend/ml-service && pip install -r requirements.txt
    
    - name: Run performance tests
      run: |
        # Start services
        npm run dev &
        cd backend/ml-service && python app.py &
        sleep 10
        
        # Run benchmarks
        node scripts/benchmark.js > benchmark-results.json
        
    - name: Compare with baseline
      run: |
        if [ -f benchmark-baseline.json ]; then
          node scripts/compare-benchmarks.js benchmark-baseline.json benchmark-results.json
        fi
        
    - name: Comment PR with results
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('benchmark-results.json'));
          const comment = `## Performance Benchmark Results
          
          | Metric | Result | Target | Status |
          |--------|--------|--------|--------|
          | Load Time | ${results.loadTime}ms | <2000ms | ${results.loadTime < 2000 ? '✅' : '❌'} |
          | Trading Cycle | ${results.cycleTime}ms | <45000ms | ${results.cycleTime < 45000 ? '✅' : '❌'} |
          | Memory Usage | ${results.memory}MB | <250MB | ${results.memory < 250 ? '✅' : '❌'} |
          `;
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

### 3. Phase Completion Workflow
Create `.github/workflows/phase-completion.yml`:

```yaml
name: Phase Completion Check

on:
  pull_request:
    types: [opened, edited]
    branches: [ main ]

jobs:
  check-phase:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Extract phase number
      id: phase
      run: |
        branch="${{ github.head_ref }}"
        phase_num=$(echo $branch | grep -oP 'phase-\K\d+')
        echo "phase=$phase_num" >> $GITHUB_OUTPUT
    
    - name: Check deliverables
      run: |
        phase=${{ steps.phase.outputs.phase }}
        python scripts/check_deliverables.py --phase $phase
    
    - name: Generate completion report
      run: |
        phase=${{ steps.phase.outputs.phase }}
        echo "# Phase $phase Completion Report" > completion-report.md
        echo "" >> completion-report.md
        
        # Add checklist based on phase
        case $phase in
          1)
            echo "## Phase 1 Deliverables" >> completion-report.md
            echo "- [x] Frontend Application" >> completion-report.md
            echo "- [x] Trading Logic" >> completion-report.md
            echo "- [x] API Integration" >> completion-report.md
            echo "- [x] ML Foundation" >> completion-report.md
            ;;
          2)
            echo "## Phase 2 Deliverables" >> completion-report.md
            echo "- [ ] Backend Service" >> completion-report.md
            echo "- [ ] Data Pipeline" >> completion-report.md
            echo "- [ ] Testing Infrastructure" >> completion-report.md
            ;;
        esac
        
    - name: Add label
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.addLabels({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            labels: ['phase-${{ steps.phase.outputs.phase }}']
          });
```

---

## Pull Request Templates

### Phase Completion PR Template
Create `.github/pull_request_template/phase_completion.md`:

```markdown
## Phase [X] Completion

### Phase Overview
- **Phase Number:** [1-5]
- **Completion Percentage:** [XX%]
- **Sprint Duration:** [X weeks]

### Deliverables Completed
- [ ] Deliverable 1
- [ ] Deliverable 2
- [ ] Deliverable 3

### SOW Requirements Addressed
| Requirement | Completion | Notes |
|------------|------------|-------|
| Requirement 1 | XX% | |
| Requirement 2 | XX% | |

### Testing Performed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Manual testing completed

### Performance Metrics
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Execution Time | | | |
| Memory Usage | | | |
| Success Rate | | | |

### Known Issues
- Issue 1: [Description]
- Issue 2: [Description]

### Documentation Updates
- [ ] README updated
- [ ] API documentation
- [ ] Testing guide
- [ ] Deployment guide

### Next Phase Prerequisites
- [ ] All tests passing
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Security review (if applicable)

### Reviewer Checklist
- [ ] Code quality acceptable
- [ ] Tests adequate
- [ ] Documentation sufficient
- [ ] Performance acceptable
- [ ] Security considerations addressed
```

---

## Issue Templates

### 1. Phase Bug Report
Create `.github/ISSUE_TEMPLATE/phase_bug_report.md`:

```markdown
---
name: Phase Bug Report
about: Report a bug found during phase testing
title: '[PHASE-X] Bug: '
labels: bug, phase-X
assignees: ''
---

**Phase:** [1-5]
**Component:** [Frontend/Backend/ML/Integration]

**Description:**
Clear description of the bug

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Impact on Phase Completion:**
- [ ] Blocks phase completion
- [ ] Major impact
- [ ] Minor impact

**Environment:**
- OS: [e.g., macOS, Windows, Linux]
- Node version: [e.g., 18.x]
- Browser: [e.g., Chrome, Firefox]

**Additional Context:**
Any other relevant information
```

### 2. Phase Feature Request
Create `.github/ISSUE_TEMPLATE/phase_feature.md`:

```markdown
---
name: Phase Feature Request
about: Suggest a feature for a specific phase
title: '[PHASE-X] Feature: '
labels: enhancement, phase-X
assignees: ''
---

**Phase:** [1-5]
**Priority:** [P0/P1/P2]

**Feature Description:**
Clear description of the proposed feature

**SOW Alignment:**
How this aligns with SOW requirements

**Implementation Approach:**
Suggested implementation

**Testing Approach:**
How to validate this feature

**Dependencies:**
- [ ] Dependency 1
- [ ] Dependency 2
```

---

## Labels Configuration

Create these labels in your repository:

```yaml
# Phase labels
- name: phase-1
  color: 0E8A16
  description: Phase 1 - Foundation & Core Trading
  
- name: phase-2
  color: 1D76DB
  description: Phase 2 - Backend Services
  
- name: phase-3
  color: 5319E7
  description: Phase 3 - Reinforcement Learning
  
- name: phase-4
  color: B60205
  description: Phase 4 - Institutional Strategies
  
- name: phase-5
  color: FBCA04
  description: Phase 5 - Production Optimization

# Status labels
- name: phase-complete
  color: 0E8A16
  description: Phase completed and approved
  
- name: phase-in-progress
  color: 1D76DB
  description: Phase currently being worked on
  
- name: phase-blocked
  color: B60205
  description: Phase blocked by dependencies

# Priority labels
- name: P0-critical
  color: B60205
  description: Critical for phase completion
  
- name: P1-important
  color: FBCA04
  description: Important but not blocking
  
- name: P2-nice-to-have
  color: C5DEF5
  description: Nice to have enhancement
```

---

## Milestone Configuration

Create milestones for each phase:

```yaml
Milestone: Phase 1 - Foundation
Due date: [Date]
Description: |
  Core trading system with React frontend
  - Trading dashboard
  - Alpaca integration
  - Basic ML service
  Completion: 50% of project

Milestone: Phase 2 - Backend Infrastructure
Due date: [Date + 2 weeks]
Description: |
  24/7 backend services
  - Node.js backend
  - WebSocket streaming
  - Data pipeline
  Completion: 65% of project

Milestone: Phase 3 - Reinforcement Learning
Due date: [Date + 5 weeks]
Description: |
  RL trading intelligence
  - PPO agent
  - Online learning
  - Strategy adaptation
  Completion: 80% of project

Milestone: Phase 4 - Institutional Strategies
Due date: [Date + 7 weeks]
Description: |
  Advanced trading strategies
  - Liquidity hunting
  - Smart money tracking
  - Multi-agent system
  Completion: 90% of project

Milestone: Phase 5 - Production
Due date: [Date + 9 weeks]
Description: |
  Production optimization
  - Sub-100ms execution
  - Binance migration
  - Live trading ready
  Completion: 100% of project
```

---

## AI Agent Instructions

### For CTO's AI Agents

Add this to your repository README or a dedicated AI_INSTRUCTIONS.md:

```markdown
## AI Agent Review Instructions

### Quick Analysis Commands

1. **Analyze Phase Status**
```bash
git log --oneline --graph --branches=phase-*
git diff main..phase-2 --stat
```

2. **Check Code Quality**
```bash
npm run lint
npx tsc --noEmit
npm audit
```

3. **Review Performance**
```bash
npm run build -- --stats
node scripts/analyze-bundle.js
```

4. **Validate Deliverables**
```bash
python scripts/check_deliverables.py --phase 1
grep -r "TODO\|FIXME\|XXX" src/
```

### Key Files to Review

| Phase | Priority Files |
|-------|---------------|
| 1 | src/services/tradingAgent.ts, src/App.tsx |
| 2 | backend/src/services/*, docker-compose.yml |
| 3 | backend/ml-service/rl_agent.py |
| 4 | backend/src/strategies/* |
| 5 | infrastructure/*, deployment/* |

### Automated Review Checklist

- [ ] All tests passing
- [ ] No TypeScript errors
- [ ] Bundle size < 5MB
- [ ] API response time < 500ms
- [ ] Memory usage < 500MB
- [ ] Documentation updated
- [ ] Security vulnerabilities: 0 high, 0 critical
```

---

## Release Process

### Phase Release Checklist

```bash
#!/bin/bash
# scripts/release-phase.sh

PHASE=$1

echo "Releasing Phase $PHASE..."

# 1. Run all checks
npm run lint
npm run build
npm test

# 2. Update version
npm version minor -m "Release Phase $PHASE"

# 3. Tag release
git tag -a "v0.${PHASE}0.0-phase${PHASE}" -m "Phase $PHASE Complete"

# 4. Generate changelog
git log --pretty=format:"- %s" v0.$((PHASE-1))0.0-phase$((PHASE-1))..HEAD > CHANGELOG-phase${PHASE}.md

# 5. Push to repository
git push origin phase-${PHASE}
git push origin --tags

# 6. Create GitHub release
gh release create "v0.${PHASE}0.0-phase${PHASE}" \
  --title "Phase $PHASE Complete" \
  --notes-file CHANGELOG-phase${PHASE}.md \
  --prerelease

echo "Phase $PHASE released successfully!"
```

---

## Monitoring and Metrics

### GitHub Insights Configuration

Enable these insights:
- Dependency graph
- Security alerts
- Code scanning
- Secret scanning

### Project Board Setup

Create a project board with columns:
- Backlog
- Phase 1 (Complete)
- Phase 2 (In Progress)
- Phase 3 (Planned)
- Phase 4 (Planned)
- Phase 5 (Planned)

### Success Metrics Tracking

Track these metrics per phase:
- Completion percentage
- Open issues
- Test coverage
- Performance benchmarks
- Documentation completeness

---

This GitHub configuration ensures structured phase-based development with comprehensive review processes suitable for both human and AI agent evaluation.