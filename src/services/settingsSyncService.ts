import React from 'react';
import { statePersistenceService } from './persistence/statePersistenceService';
import { auditLogService } from './persistence/auditLogService';

export interface SettingsUpdate {
  id: string;
  timestamp: number;
  settings: any;
  applied: boolean;
}

class SettingsSyncService {
  private updateQueue: SettingsUpdate[] = [];
  private isProcessing: boolean = false;
  private listeners: Set<(update: SettingsUpdate) => void> = new Set();
  private nextCycleTime: number = Date.now() + 45000; // 45 seconds
  private cycleTimer: number | null = null;

  constructor() {
    this.startCycleTimer();
  }

  /**
   * Queue a settings update for the next trading cycle
   */
  queueUpdate(settings: any): string {
    const updateId = `update_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const update: SettingsUpdate = {
      id: updateId,
      timestamp: Date.now(),
      settings,
      applied: false
    };
    
    this.updateQueue.push(update);
    
    // Notify listeners
    this.notifyListeners(update);
    
    // Log the queued update
    console.log(`Settings update ${updateId} queued for next cycle`);
    
    return updateId;
  }

  /**
   * Get pending updates
   */
  getPendingUpdates(): SettingsUpdate[] {
    return this.updateQueue.filter(u => !u.applied);
  }

  /**
   * Apply pending settings at the start of a new cycle
   */
  async applyPendingSettings(): Promise<void> {
    if (this.isProcessing || this.updateQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    try {
      const pendingUpdates = this.getPendingUpdates();
      
      if (pendingUpdates.length === 0) {
        return;
      }
      
      // Get the most recent update (last one wins)
      const latestUpdate = pendingUpdates[pendingUpdates.length - 1];
      
      // Apply the settings
      await statePersistenceService.saveUserSettings(latestUpdate.settings);
      
      // Log the application
      await auditLogService.logSettingChange(
        'settings_sync',
        null,
        latestUpdate.settings,
        `Applied queued settings update ${latestUpdate.id}`
      );
      
      // Mark all pending updates as applied
      pendingUpdates.forEach(update => {
        update.applied = true;
      });
      
      // Clear old updates from queue (keep last 10 for history)
      if (this.updateQueue.length > 10) {
        this.updateQueue = this.updateQueue.slice(-10);
      }
      
      console.log(`Applied settings update ${latestUpdate.id}`);
      
      // Notify listeners that settings were applied
      this.notifyListeners(latestUpdate);
      
    } catch (error) {
      console.error('Failed to apply pending settings:', error);
      await auditLogService.logSystemAlert(
        'settings_sync_error',
        'Failed to apply pending settings',
        { error: error.message }
      );
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Get the time until next cycle
   */
  getTimeUntilNextCycle(): number {
    return Math.max(0, this.nextCycleTime - Date.now());
  }

  /**
   * Get next cycle time as a formatted string
   */
  getNextCycleTimeString(): string {
    const seconds = Math.ceil(this.getTimeUntilNextCycle() / 1000);
    if (seconds <= 0) {
      return 'Now';
    }
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  }

  /**
   * Subscribe to settings updates
   */
  subscribe(callback: (update: SettingsUpdate) => void): () => void {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  /**
   * Check if there are pending updates
   */
  hasPendingUpdates(): boolean {
    return this.getPendingUpdates().length > 0;
  }

  /**
   * Cancel a pending update
   */
  cancelUpdate(updateId: string): boolean {
    const index = this.updateQueue.findIndex(u => u.id === updateId && !u.applied);
    if (index !== -1) {
      this.updateQueue.splice(index, 1);
      console.log(`Cancelled settings update ${updateId}`);
      return true;
    }
    return false;
  }

  /**
   * Force apply settings immediately (for critical updates)
   */
  async forceApplyNow(settings: any): Promise<void> {
    try {
      await statePersistenceService.saveUserSettings(settings);
      
      await auditLogService.logSettingChange(
        'settings_sync',
        null,
        settings,
        'Force applied settings (immediate)'
      );
      
      console.log('Settings force applied immediately');
    } catch (error) {
      console.error('Failed to force apply settings:', error);
      throw error;
    }
  }

  // Private methods
  private notifyListeners(update: SettingsUpdate): void {
    this.listeners.forEach(callback => {
      try {
        callback(update);
      } catch (error) {
        console.error('Error in settings update listener:', error);
      }
    });
  }

  private startCycleTimer(): void {
    // Update next cycle time every 45 seconds
    this.cycleTimer = window.setInterval(() => {
      this.nextCycleTime = Date.now() + 45000;
      
      // Apply pending settings at the start of each cycle
      this.applyPendingSettings();
    }, 45000);
  }

  /**
   * Clean up
   */
  destroy(): void {
    if (this.cycleTimer) {
      clearInterval(this.cycleTimer);
    }
    this.listeners.clear();
    this.updateQueue = [];
  }
}

// Export singleton instance
export const settingsSyncService = new SettingsSyncService();

// Hook for React components
export const useSettingsSync = () => {
  const [hasPending, setHasPending] = React.useState(false);
  const [nextCycle, setNextCycle] = React.useState('');
  
  React.useEffect(() => {
    // Check for pending updates
    const checkPending = () => {
      setHasPending(settingsSyncService.hasPendingUpdates());
      setNextCycle(settingsSyncService.getNextCycleTimeString());
    };
    
    // Initial check
    checkPending();
    
    // Subscribe to updates
    const unsubscribe = settingsSyncService.subscribe(() => {
      checkPending();
    });
    
    // Update timer
    const timer = setInterval(checkPending, 1000);
    
    return () => {
      unsubscribe();
      clearInterval(timer);
    };
  }, []);
  
  return {
    hasPending,
    nextCycle,
    queueUpdate: settingsSyncService.queueUpdate.bind(settingsSyncService),
    cancelUpdate: settingsSyncService.cancelUpdate.bind(settingsSyncService),
    forceApplyNow: settingsSyncService.forceApplyNow.bind(settingsSyncService)
  };
};