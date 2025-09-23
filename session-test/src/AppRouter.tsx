import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthModal } from './components/Auth/AuthModal';
import { useAuth } from './hooks/useAuth';
import { Dashboard } from './pages/Dashboard';
import { AgentControlsEnhanced } from './pages/AgentControlsEnhanced';
import { AuditLogs } from './pages/AuditLogs';
import { Bitcoin } from 'lucide-react';
import { useState } from 'react';

function AppRouter() {
  const { user, loading: authLoading } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="mb-8">
            <Bitcoin className="h-16 w-16 text-orange-400 mx-auto mb-4" />
            <h1 className="text-3xl font-bold text-white mb-2">AI Crypto Trading Agent</h1>
            <p className="text-gray-400">Sign in to access your $50K virtual portfolio</p>
          </div>
          <button
            onClick={() => setShowAuthModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors"
          >
            Get Started
          </button>
        </div>
        <AuthModal
          isOpen={showAuthModal}
          onClose={() => setShowAuthModal(false)}
          onAuthSuccess={() => setShowAuthModal(false)}
        />
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/agent-controls" element={<AgentControlsEnhanced />} />
        <Route path="/audit-logs" element={<AuditLogs />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default AppRouter;