import React, { useState, useEffect } from 'react';
import { Responsive, WidthProvider, Layout } from 'react-grid-layout';
import { GripVertical, Settings, RotateCcw } from 'lucide-react';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

interface DashboardWidget {
  id: string;
  title: string;
  component: React.ReactNode;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
}

interface DraggableGridProps {
  widgets: DashboardWidget[];
  onLayoutChange?: (layout: Layout[]) => void;
}

export const DraggableGrid: React.FC<DraggableGridProps> = ({ widgets, onLayoutChange }) => {
  const [isEditMode, setIsEditMode] = useState(false);
  const [layouts, setLayouts] = useState<{ [key: string]: Layout[] }>({});
  const [currentBreakpoint, setCurrentBreakpoint] = useState('lg');

  // Default layout configuration with proper spacing
  const getDefaultLayout = (): Layout[] => {
    return [
      { i: 'account-summary', x: 0, y: 0, w: 12, h: 4, minW: 6, minH: 3, static: false },
      { i: 'positions', x: 0, y: 4, w: 6, h: 8, minW: 4, minH: 6, static: false },
      { i: 'watchlist', x: 6, y: 4, w: 6, h: 8, minW: 4, minH: 6, static: false },
      { i: 'orders', x: 0, y: 12, w: 8, h: 7, minW: 6, minH: 5, static: false },
      { i: 'fear-greed', x: 8, y: 12, w: 4, h: 7, minW: 3, minH: 5, static: false },
      { i: 'trading-controls', x: 0, y: 19, w: 6, h: 10, minW: 4, minH: 8, static: false },
      { i: 'trading-signals', x: 6, y: 19, w: 6, h: 10, minW: 4, minH: 8, static: false },
      { i: 'auto-trade-activity', x: 0, y: 29, w: 12, h: 24, minW: 10, minH: 16, static: false },
      { i: 'auto-trade-settings', x: 0, y: 53, w: 6, h: 12, minW: 4, minH: 8, static: false },
      { i: 'market-insights', x: 0, y: 65, w: 12, h: 52, minW: 10, minH: 38, static: false },
    ];
  };

  // Function to check if two items overlap
  const itemsOverlap = (item1: Layout, item2: Layout): boolean => {
    return !(
      item1.x + item1.w <= item2.x ||
      item2.x + item2.w <= item1.x ||
      item1.y + item1.h <= item2.y ||
      item2.y + item2.h <= item1.y
    );
  };

  // Function to find next available position
  const findAvailablePosition = (layout: Layout[], item: Layout, cols: number): Layout => {
    const newItem = { ...item };
    
    // Try to find a position that doesn't overlap
    for (let y = 0; y < 100; y++) {
      for (let x = 0; x <= cols - newItem.w; x++) {
        const testItem = { ...newItem, x, y };
        const hasOverlap = layout.some(existingItem => 
          existingItem.i !== testItem.i && itemsOverlap(testItem, existingItem)
        );
        
        if (!hasOverlap) {
          return testItem;
        }
      }
    }
    
    return newItem;
  };

  // Function to resolve overlaps in layout
  const resolveOverlaps = (layout: Layout[], cols: number): Layout[] => {
    const resolvedLayout: Layout[] = [];
    
    // Sort by y position first, then x position to maintain visual order
    const sortedLayout = [...layout].sort((a, b) => {
      if (a.y === b.y) return a.x - b.x;
      return a.y - b.y;
    });
    
    sortedLayout.forEach(item => {
      const resolvedItem = findAvailablePosition(resolvedLayout, item, cols);
      resolvedLayout.push(resolvedItem);
    });
    
    return resolvedLayout;
  };

  // Function to compact layout vertically while preventing overlaps
  const compactLayout = (layout: Layout[], cols: number): Layout[] => {
    const compactedLayout = [...layout];
    
    // Sort by y position
    compactedLayout.sort((a, b) => a.y - b.y);
    
    // Move each item up as much as possible
    compactedLayout.forEach((item, index) => {
      let newY = 0;
      
      // Check against all previous items
      for (let i = 0; i < index; i++) {
        const otherItem = compactedLayout[i];
        if (!(item.x + item.w <= otherItem.x || otherItem.x + otherItem.w <= item.x)) {
          // Items overlap horizontally, so this item must be below the other
          newY = Math.max(newY, otherItem.y + otherItem.h);
        }
      }
      
      item.y = newY;
    });
    
    return compactedLayout;
  };

  useEffect(() => {
    const defaultLayout = getDefaultLayout();
    const cols = { lg: 12, md: 10, sm: 8, xs: 4 };
    
    let defaultLayouts = {
      lg: compactLayout(defaultLayout, cols.lg),
      md: compactLayout(defaultLayout.map(item => ({ 
        ...item, 
        w: Math.min(item.w, 10),
        x: item.x > 6 ? Math.max(0, item.x - 2) : item.x 
      })), cols.md),
      sm: compactLayout(defaultLayout.map(item => ({ 
        ...item, 
        w: Math.min(item.w, 8),
        x: item.x > 4 ? 0 : item.x,
      })), cols.sm),
      xs: defaultLayout.map((item, index) => ({ 
        ...item, 
        w: 4,
        h: item.i === 'market-insights' ? 40 : item.i === 'auto-trade-activity' ? Math.max(item.h, 22) : item.i === 'whale-alerts' ? Math.max(item.h, 22) : item.h,
        x: 0,
        y: index * (item.h + 1)
      })),
    };
    
    setLayouts(defaultLayouts);
    
    // Load saved layout after initial render
    setTimeout(() => {
      const savedLayouts = localStorage.getItem('dashboard-layouts');
      if (savedLayouts) {
        try {
          const parsed = JSON.parse(savedLayouts);
          // Ensure new widgets exist in saved layout; append if missing
          const defaults = getDefaultLayout();
          const ensureAll = (items: Layout[], bpCols: number) => {
            const existingIds = new Set(items.map(i => i.i));
            const toAdd = defaults.filter(d => !existingIds.has(d.i));
            let merged = [...items];
            for (const add of toAdd) {
              merged.push(findAvailablePosition(merged, { ...add }, bpCols));
            }
            return resolveOverlaps(merged, bpCols);
          };

          const mergedLayouts = { ...parsed };
          mergedLayouts.lg = ensureAll(parsed.lg || [], cols.lg);
          mergedLayouts.md = ensureAll(parsed.md || [], cols.md);
          mergedLayouts.sm = ensureAll(parsed.sm || [], cols.sm);
          mergedLayouts.xs = ensureAll(parsed.xs || [], cols.xs);

          setLayouts(mergedLayouts);
        } catch (error) {
          console.error('Error parsing saved layouts:', error);
          setLayouts(defaultLayouts);
        }
      }
    }, 100);
  }, []);

  const handleLayoutChange = (layout: Layout[], allLayouts: { [key: string]: Layout[] }) => {
    // Only update layouts in edit mode and when user is actively dragging/resizing
    if (isEditMode) {
      setLayouts(allLayouts);
      localStorage.setItem('dashboard-layouts', JSON.stringify(allLayouts));
    }
    
    if (onLayoutChange) {
      onLayoutChange(layout);
    }
  };

  const handleBreakpointChange = (breakpoint: string) => {
    setCurrentBreakpoint(breakpoint);
  };

  const handleDragStop = (layout: Layout[]) => {
    if (!isEditMode) return;
    
    const cols = { lg: 12, md: 10, sm: 8, xs: 4 };
    const resolvedLayout = resolveOverlaps(layout, cols[currentBreakpoint as keyof typeof cols] || 12);
    
    const newLayouts = { ...layouts };
    newLayouts[currentBreakpoint] = resolvedLayout;
    
    setLayouts(newLayouts);
    localStorage.setItem('dashboard-layouts', JSON.stringify(newLayouts));
  };

  const handleResizeStop = (layout: Layout[]) => {
    if (!isEditMode) return;
    
    const cols = { lg: 12, md: 10, sm: 8, xs: 4 };
    const resolvedLayout = resolveOverlaps(layout, cols[currentBreakpoint as keyof typeof cols] || 12);
    
    const newLayouts = { ...layouts };
    newLayouts[currentBreakpoint] = resolvedLayout;
    
    setLayouts(newLayouts);
    localStorage.setItem('dashboard-layouts', JSON.stringify(newLayouts));
  };

  const resetLayout = () => {
    const defaultLayout = getDefaultLayout();
    const cols = { lg: 12, md: 10, sm: 8, xs: 4 };
    
    const defaultLayouts = {
      lg: compactLayout(defaultLayout, cols.lg),
      md: compactLayout(defaultLayout.map(item => ({ 
        ...item, 
        w: Math.min(item.w, 10),
        x: item.x > 6 ? Math.max(0, item.x - 2) : item.x 
      })), cols.md),
      sm: compactLayout(defaultLayout.map(item => ({ 
        ...item, 
        w: Math.min(item.w, 8),
        x: item.x > 4 ? 0 : item.x,
      })), cols.sm),
      xs: defaultLayout.map((item, index) => ({ 
        ...item, 
        w: 4, 
        x: 0,
        y: index * (item.h + 1)
      })),
    };
    
    setLayouts(defaultLayouts);
    localStorage.setItem('dashboard-layouts', JSON.stringify(defaultLayouts));
  };

  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480 };

  return (
    <div className="relative">
      {/* Dashboard Controls */}
      <div className="flex items-center justify-between mb-6 bg-gray-800 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-bold text-white">Dashboard</h2>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isEditMode 
              ? 'bg-orange-900/30 text-orange-400 border border-orange-400' 
              : 'bg-gray-700 text-gray-300'
          }`}>
            {isEditMode ? 'Edit Mode' : 'View Mode'}
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={resetLayout}
            className="flex items-center px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm"
            title="Reset to default layout"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </button>
          <button
            onClick={() => setIsEditMode(!isEditMode)}
            className={`flex items-center px-4 py-2 rounded-lg transition-colors font-medium ${
              isEditMode
                ? 'bg-orange-600 hover:bg-orange-700 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            <Settings className="h-4 w-4 mr-2" />
            {isEditMode ? 'Done' : 'Customize'}
          </button>
        </div>
      </div>

      {/* Grid Layout */}
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        onLayoutChange={handleLayoutChange}
        onBreakpointChange={handleBreakpointChange}
        onDragStop={handleDragStop}
        onResizeStop={handleResizeStop}
        breakpoints={breakpoints}
        cols={{ lg: 12, md: 10, sm: 8, xs: 4 }}
        rowHeight={40}
        isDraggable={isEditMode}
        isResizable={isEditMode}
        margin={[12, 12]}
        containerPadding={[0, 0]}
        useCSSTransforms={true}
        preventCollision={false}
        allowOverlap={true}
        compactType={null}
        verticalCompact={false}
        autoSize={true}
        isBounded={true}
        resizeHandles={['se']}
        draggableHandle=".drag-handle"
      >
        {widgets.map((widget) => (
          <div key={widget.id} className="relative bg-gray-800 rounded-lg overflow-hidden">
            {/* Widget Header (only visible in edit mode) */}
            {isEditMode && (
              <div className="drag-handle absolute top-0 left-0 right-0 bg-gray-700/95 backdrop-blur-sm border-b border-gray-600 px-3 py-2 flex items-center justify-between z-10 rounded-t-lg cursor-move">
                <div className="flex items-center">
                  <GripVertical className="h-4 w-4 text-gray-400 mr-2" />
                  <span className="text-white text-sm font-medium">{widget.title}</span>
                </div>
                <div className="text-xs text-gray-400">
                  Drag to move • Resize from corners
                </div>
              </div>
            )}
            
            {/* Widget Content */}
            <div className={`h-full ${isEditMode ? 'pt-10' : ''} transition-all duration-200`}>
              <div className={`h-full ${isEditMode ? 'opacity-80 hover:opacity-100' : ''} transition-opacity`}>
                {widget.component}
              </div>
            </div>

            {/* Edit Mode Overlay */}
            {isEditMode && (
              <div className="absolute inset-0 border-2 border-dashed border-blue-400/50 rounded-lg pointer-events-none" />
            )}
          </div>
        ))}
      </ResponsiveGridLayout>

      {/* Edit Mode Instructions removed per user request */}
    </div>
  );
};