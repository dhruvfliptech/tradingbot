# Architecture Decisions Record (ADR)

## Critical Architecture Constraints

### ADR-001: Frontend Framework - React + Vite (NOT Next.js)

**Status:** DECIDED AND LOCKED

**Decision:** This project uses React 18 with Vite 5.4.14 as the build tool.

**Context:**
- Project was initialized with React + Vite
- All existing code is built on React + Vite architecture
- Phase 1 (50% of project) is complete with this stack

**Constraints:**
- **NEVER use Next.js for this project**
- **NEVER suggest migrating to Next.js**
- **DO NOT implement Next.js features (SSR, SSG, API routes)**
- **USE React Router for routing**
- **USE Vite for building and development**

**Rationale:**
1. Project is already 50% complete with React + Vite
2. Migration would require complete rewrite
3. React + Vite provides necessary performance for trading bot
4. Client requirements are met with current stack
5. No SSR/SSG requirements that would justify Next.js

**Consequences:**
- All routing must use React Router
- API calls go to separate backend services
- No server-side rendering
- Build process uses Vite bundler
- Development server uses Vite HMR

---

## Architectural Overview

### System Architecture: Microservices

**Components:**
1. **Frontend (React + Vite)**
   - Dashboard UI
   - Real-time monitoring
   - Configuration management
   - Built with: React 18, Vite 5.4.14, TypeScript, Tailwind CSS

2. **Backend Services**
   - **Node.js/Express Service** (Port 3001)
     - Main API gateway
     - Trading orchestration
     - Risk management
     - Database operations
   
   - **Python Flask Service** (Port 5000)
     - ML model serving
     - Backtesting engine
     - Data analysis
     - Composer MCP integration

3. **Database Layer**
   - Supabase (PostgreSQL)
   - Future: Redis cache (Phase 2)

4. **External Integrations**
   - Trading: Alpaca (current), Binance (Phase 2)
   - Data: 6 free crypto APIs
   - AI: Groq for sentiment
   - Backtesting: Composer MCP

---

## Technology Stack Decisions

### ADR-002: Backend Technology

**Decision:** Dual backend with Node.js and Python

**Rationale:**
- Node.js for real-time trading and API management
- Python for ML/data science capabilities
- Best of both ecosystems

### ADR-003: Database Choice

**Decision:** Supabase (managed PostgreSQL)

**Rationale:**
- Managed service reduces operational overhead
- Built-in auth and real-time capabilities
- PostgreSQL for complex queries
- Good free tier for development

### ADR-004: ML Architecture

**Decision:** AdaptiveThreshold pre-RL system

**Rationale:**
- Simpler than full RL implementation
- Faster to train and deploy
- Provides adaptive learning
- Good baseline for Phase 1

### ADR-005: Trading Exchange Priority

**Decision:** Alpaca first, Binance later

**Updated Context:**
- Client needs more time for Binance setup
- Alpaca provides good testing environment
- Similar API patterns for easy migration

---

## Development Phases

### Phase 1 (COMPLETE - 50%)
- ✅ React + Vite frontend setup
- ✅ Dual backend services (Node.js + Python)
- ✅ Basic trading logic with Alpaca
- ✅ AdaptiveThreshold ML system
- ✅ Risk management implementation
- ✅ 6 free API integrations
- ✅ Composer MCP backtesting

### Phase 2 (In Progress - 25%)
- API key configuration
- Enhanced ML models
- Advanced strategies
- Real-time monitoring

### Phase 3 (Planned - 25%)
- Binance integration
- Production deployment
- Performance optimization
- Comprehensive testing

---

## Critical Implementation Guidelines

### For All Developers/Agents:

1. **Frontend Development**
   ```bash
   # CORRECT - Use these commands:
   npm run dev        # Start Vite dev server
   npm run build      # Build with Vite
   npm run preview    # Preview production build
   
   # WRONG - Never use these:
   next dev           # NO Next.js
   next build         # NO Next.js
   next start         # NO Next.js
   ```

2. **Routing**
   ```javascript
   // CORRECT - Use React Router:
   import { BrowserRouter, Routes, Route } from 'react-router-dom';
   
   // WRONG - Never use Next.js routing:
   import { useRouter } from 'next/router'; // NO
   import Link from 'next/link';            // NO
   ```

3. **API Routes**
   ```javascript
   // CORRECT - Call backend services:
   fetch('http://localhost:3001/api/trades')
   
   // WRONG - No Next.js API routes:
   // pages/api/trades.js  // NO
   ```

4. **File Structure**
   ```
   CORRECT Structure:
   /src
     /components    # React components
     /pages        # React page components
     /services     # API services
     /hooks        # Custom React hooks
   
   WRONG Structure:
   /pages         # Next.js pages directory
   /app           # Next.js app directory
   /api           # Next.js API routes
   ```

---

## Performance Requirements

- **Frontend Load Time:** < 3 seconds
- **API Response Time:** < 100ms
- **Trade Execution:** < 100ms
- **Dashboard Update Rate:** Real-time (WebSocket)
- **Uptime Target:** 99.9%

---

## Security Considerations

- API keys stored in environment variables
- Never commit sensitive data
- Use HTTPS in production
- Implement rate limiting
- Validate all user inputs
- Secure WebSocket connections

---

## Deployment Architecture

### Development
- Frontend: Vite dev server (port 5173)
- Backend: Node.js (port 3001), Python (port 5000)
- Database: Supabase cloud

### Production (Phase 3)
- Frontend: Static hosting (Vercel/Netlify)
- Backend: Cloud services (AWS/GCP)
- Database: Supabase cloud
- Monitoring: Custom dashboard + logs

---

## Decision Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|---------|
| Project Start | React + Vite | Fast, modern, no SSR needed | Permanent |
| Phase 1 | Microservices | Separation of concerns | High |
| Phase 1 | Alpaca first | Easier setup, testing | Medium |
| Phase 1 | AdaptiveThreshold | Simpler than full RL | Medium |
| Aug 15 | Defer Binance | Client needs time | Low |

---

## Contact for Questions

**Project Lead:** Jay Kinney (Flip-Tech Inc)
**Client:** Damiano Duran (Core Calling LLC)

---

*This document is the source of truth for all architectural decisions.*
*Any deviation from these decisions requires explicit approval.*
*Last Updated: August 15, 2025*