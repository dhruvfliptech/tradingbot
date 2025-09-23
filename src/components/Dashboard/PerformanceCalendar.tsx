import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, TrendingUp, TrendingDown } from 'lucide-react';
import { virtualPortfolioService } from '../../services/persistence/virtualPortfolioService';

interface DayData {
  date: string;
  pnl: number;
  pnlPercent: number;
  trades: number;
}

interface PerformanceCalendarProps {
  className?: string;
}

export const PerformanceCalendar: React.FC<PerformanceCalendarProps> = ({ className = '' }) => {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [dayData, setDayData] = useState<Map<string, DayData>>(new Map());
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState<'1D' | '1W' | '1M' | '3M' | 'YTD'>('1M');
  const [hoveredDay, setHoveredDay] = useState<string | null>(null);

  useEffect(() => {
    loadPerformanceData();
  }, [currentMonth, selectedPeriod]);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      
      // Calculate days based on period
      let days = 30;
      switch (selectedPeriod) {
        case '1D': days = 1; break;
        case '1W': days = 7; break;
        case '1M': days = 30; break;
        case '3M': days = 90; break;
        case 'YTD': 
          const now = new Date();
          const yearStart = new Date(now.getFullYear(), 0, 1);
          days = Math.floor((now.getTime() - yearStart.getTime()) / (1000 * 60 * 60 * 24));
          break;
      }
      
      const snapshots = await virtualPortfolioService.getDailySnapshots(days);
      
      const dataMap = new Map<string, DayData>();
      snapshots.forEach(snapshot => {
        dataMap.set(snapshot.snapshot_date, {
          date: snapshot.snapshot_date,
          pnl: snapshot.daily_pnl,
          pnlPercent: snapshot.daily_pnl_percent,
          trades: snapshot.trades_count
        });
      });
      
      setDayData(dataMap);
    } catch (error) {
      console.error('Failed to load performance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDaysInMonth = (date: Date) => {
    return new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
  };

  const getFirstDayOfMonth = (date: Date) => {
    return new Date(date.getFullYear(), date.getMonth(), 1).getDay();
  };

  const formatMonth = (date: Date) => {
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  };

  const navigateMonth = (direction: 'prev' | 'next') => {
    setCurrentMonth(prev => {
      const newDate = new Date(prev);
      if (direction === 'prev') {
        newDate.setMonth(newDate.getMonth() - 1);
      } else {
        newDate.setMonth(newDate.getMonth() + 1);
      }
      return newDate;
    });
  };

  const getHeatmapColor = (pnlPercent: number) => {
    if (pnlPercent === 0) return 'bg-gray-700';
    
    const absPercent = Math.abs(pnlPercent);
    
    if (pnlPercent > 0) {
      // Green shades for profit
      if (absPercent < 1) return 'bg-green-900/50';
      if (absPercent < 2) return 'bg-green-800/70';
      if (absPercent < 3) return 'bg-green-700';
      if (absPercent < 5) return 'bg-green-600';
      return 'bg-green-500';
    } else {
      // Red shades for loss
      if (absPercent < 1) return 'bg-red-900/50';
      if (absPercent < 2) return 'bg-red-800/70';
      if (absPercent < 3) return 'bg-red-700';
      if (absPercent < 5) return 'bg-red-600';
      return 'bg-red-500';
    }
  };

  const renderCalendarDays = () => {
    const daysInMonth = getDaysInMonth(currentMonth);
    const firstDay = getFirstDayOfMonth(currentMonth);
    const days = [];
    
    // Empty cells for days before month starts
    for (let i = 0; i < firstDay; i++) {
      days.push(
        <div key={`empty-${i}`} className="aspect-square" />
      );
    }
    
    // Days of the month
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${currentMonth.getFullYear()}-${String(currentMonth.getMonth() + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const data = dayData.get(dateStr);
      const isToday = dateStr === new Date().toISOString().split('T')[0];
      const hasFutureData = new Date(dateStr) > new Date();
      
      days.push(
        <div
          key={day}
          className={`
            aspect-square flex flex-col items-center justify-center rounded-lg cursor-pointer
            transition-all duration-200 relative group
            ${data && !hasFutureData ? getHeatmapColor(data.pnlPercent) : 'bg-gray-800'}
            ${isToday ? 'ring-2 ring-indigo-400' : ''}
            ${hasFutureData ? 'opacity-30 cursor-not-allowed' : 'hover:ring-2 hover:ring-gray-500'}
          `}
          onMouseEnter={() => setHoveredDay(dateStr)}
          onMouseLeave={() => setHoveredDay(null)}
        >
          <span className="text-xs text-gray-300">{day}</span>
          {data && !hasFutureData && (
            <>
              <span className={`text-xs font-medium ${data.pnl >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                {data.pnl >= 0 ? '+' : ''}{data.pnlPercent.toFixed(1)}%
              </span>
              {data.trades > 0 && (
                <div className="absolute bottom-1 right-1 w-2 h-2 bg-blue-400 rounded-full" />
              )}
            </>
          )}
          
          {/* Tooltip */}
          {hoveredDay === dateStr && data && !hasFutureData && (
            <div className="absolute z-10 bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl whitespace-nowrap">
              <div className="text-xs text-gray-400 mb-1">{new Date(dateStr).toLocaleDateString()}</div>
              <div className={`text-sm font-medium ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                P&L: ${data.pnl.toFixed(2)} ({data.pnlPercent.toFixed(2)}%)
              </div>
              {data.trades > 0 && (
                <div className="text-xs text-gray-400 mt-1">Trades: {data.trades}</div>
              )}
            </div>
          )}
        </div>
      );
    }
    
    return days;
  };

  const calculatePeriodStats = () => {
    const dataArray = Array.from(dayData.values());
    if (dataArray.length === 0) {
      return { totalPnL: 0, avgDaily: 0, winDays: 0, lossDays: 0, bestDay: 0, worstDay: 0 };
    }
    
    const totalPnL = dataArray.reduce((sum, d) => sum + d.pnl, 0);
    const avgDaily = totalPnL / dataArray.length;
    const winDays = dataArray.filter(d => d.pnl > 0).length;
    const lossDays = dataArray.filter(d => d.pnl < 0).length;
    const bestDay = Math.max(...dataArray.map(d => d.pnl));
    const worstDay = Math.min(...dataArray.map(d => d.pnl));
    
    return { totalPnL, avgDaily, winDays, lossDays, bestDay, worstDay };
  };

  const stats = calculatePeriodStats();

  if (loading) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-48"></div>
          <div className="grid grid-cols-7 gap-2">
            {[...Array(35)].map((_, i) => (
              <div key={i} className="aspect-square bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg border border-gray-700 p-6 h-full overflow-y-auto ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Performance Calendar</h2>
        
        {/* Period Selector */}
        <div className="flex items-center space-x-1 bg-gray-700 rounded-lg p-1">
          {(['1D', '1W', '1M', '3M', 'YTD'] as const).map(period => (
            <button
              key={period}
              onClick={() => setSelectedPeriod(period)}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                selectedPeriod === period
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {period}
            </button>
          ))}
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-4">
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Period P&L</p>
          <p className={`text-sm font-medium ${stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${stats.totalPnL.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Daily Avg</p>
          <p className={`text-sm font-medium ${stats.avgDaily >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${stats.avgDaily.toFixed(2)}
          </p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Win Days</p>
          <p className="text-sm font-medium text-green-400">{stats.winDays}</p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Loss Days</p>
          <p className="text-sm font-medium text-red-400">{stats.lossDays}</p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Best Day</p>
          <p className="text-sm font-medium text-green-400">${stats.bestDay.toFixed(2)}</p>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <p className="text-xs text-gray-400">Worst Day</p>
          <p className="text-sm font-medium text-red-400">${stats.worstDay.toFixed(2)}</p>
        </div>
      </div>

      {/* Calendar Navigation */}
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={() => navigateMonth('prev')}
          className="p-1 text-gray-400 hover:text-white transition-colors"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <h3 className="text-white font-medium">{formatMonth(currentMonth)}</h3>
        <button
          onClick={() => navigateMonth('next')}
          className="p-1 text-gray-400 hover:text-white transition-colors"
          disabled={currentMonth.getMonth() === new Date().getMonth()}
        >
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>

      {/* Day Labels */}
      <div className="grid grid-cols-7 gap-2 mb-2">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
          <div key={day} className="text-center text-xs text-gray-500">
            {day}
          </div>
        ))}
      </div>

      {/* Calendar Grid */}
      <div className="grid grid-cols-7 gap-2">
        {renderCalendarDays()}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center space-x-4 text-xs">
        <div className="flex items-center space-x-2">
          <TrendingDown className="h-3 w-3 text-red-400" />
          <span className="text-gray-400">Loss</span>
        </div>
        <div className="flex items-center space-x-1">
          {['-5%', '-3%', '-1%', '0%', '+1%', '+3%', '+5%'].map((label, i) => (
            <div key={label} className="flex flex-col items-center">
              <div className={`w-4 h-4 rounded ${
                i === 0 ? 'bg-red-500' :
                i === 1 ? 'bg-red-700' :
                i === 2 ? 'bg-red-900/50' :
                i === 3 ? 'bg-gray-700' :
                i === 4 ? 'bg-green-900/50' :
                i === 5 ? 'bg-green-700' :
                'bg-green-500'
              }`} />
              <span className="text-gray-500 mt-1">{label}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">Profit</span>
          <TrendingUp className="h-3 w-3 text-green-400" />
        </div>
      </div>

      <style jsx>{`
        /* Custom scrollbar styling */
        .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }

        .overflow-y-auto::-webkit-scrollbar-track {
          background: #1F2937;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb {
          background: #4B5563;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background: #6B7280;
        }
      `}</style>
    </div>
  );
};