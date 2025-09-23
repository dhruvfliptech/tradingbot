import { useEffect } from 'react';
import { tradingAgentV2 } from '../services/tradingAgentV2';

/**
 * Emergency stop hook - adds Ctrl+Shift+S keyboard shortcut to stop auto-trading
 */
export const useEmergencyStop = () => {
  useEffect(() => {
    const handleKeyDown = async (event: KeyboardEvent) => {
      // Ctrl+Shift+S for emergency stop
      if (event.ctrlKey && event.shiftKey && event.key === 'S') {
        event.preventDefault();
        
        if (confirm('🚨 EMERGENCY STOP: Are you sure you want to immediately stop the auto-trading agent?')) {
          try {
            await tradingAgentV2.stop();
            alert('✅ Auto-trading agent stopped successfully!');
          } catch (error) {
            console.error('Failed to stop agent:', error);
            alert('❌ Failed to stop agent. Please try again.');
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);
};
