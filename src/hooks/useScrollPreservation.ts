import { useEffect, useRef } from 'react';

/**
 * Hook to prevent auto-scrolling when content updates
 */
export const useScrollPreservation = (dependency?: any) => {
  const scrollPositionRef = useRef<number>(0);
  const isUserScrollingRef = useRef<boolean>(false);
  const lastScrollTimeRef = useRef<number>(0);

  useEffect(() => {
    const handleScroll = () => {
      isUserScrollingRef.current = true;
      lastScrollTimeRef.current = Date.now();
      scrollPositionRef.current = window.scrollY;
      
      // Reset user scrolling flag after a delay
      setTimeout(() => {
        if (Date.now() - lastScrollTimeRef.current > 150) {
          isUserScrollingRef.current = false;
        }
      }, 200);
    };

    const handleWheel = () => {
      isUserScrollingRef.current = true;
      lastScrollTimeRef.current = Date.now();
    };

    const handleTouch = () => {
      isUserScrollingRef.current = true;
      lastScrollTimeRef.current = Date.now();
    };

    // Add scroll event listeners
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('wheel', handleWheel, { passive: true });
    window.addEventListener('touchmove', handleTouch, { passive: true });

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('wheel', handleWheel);
      window.removeEventListener('touchmove', handleTouch);
    };
  }, []);

  useEffect(() => {
    // Preserve scroll position when dependency changes
    if (dependency !== undefined) {
      const currentScroll = window.scrollY;
      
      // Only restore if user wasn't actively scrolling
      if (!isUserScrollingRef.current && 
          Date.now() - lastScrollTimeRef.current > 500) {
        requestAnimationFrame(() => {
          window.scrollTo({
            top: scrollPositionRef.current,
            behavior: 'instant'
          });
        });
      } else {
        // Update saved position if user was scrolling
        scrollPositionRef.current = currentScroll;
      }
    }
  }, [dependency]);

  const preserveScroll = (callback: () => void) => {
    const currentScroll = window.scrollY;
    callback();
    
    // Only restore if user isn't actively scrolling
    if (!isUserScrollingRef.current) {
      requestAnimationFrame(() => {
        window.scrollTo({
          top: currentScroll,
          behavior: 'instant'
        });
      });
    }
  };

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
    scrollPositionRef.current = 0;
  };

  const scrollToBottom = () => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth'
    });
    scrollPositionRef.current = document.documentElement.scrollHeight;
  };

  return {
    preserveScroll,
    scrollToTop,
    scrollToBottom,
    isUserScrolling: isUserScrollingRef.current,
    currentScroll: scrollPositionRef.current
  };
};

/**
 * Hook to maintain scroll position in a specific container
 */
export const useContainerScrollPreservation = (
  containerRef: React.RefObject<HTMLElement>,
  dependency?: any
) => {
  const scrollPositionRef = useRef<number>(0);
  const isUserScrollingRef = useRef<boolean>(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      isUserScrollingRef.current = true;
      scrollPositionRef.current = container.scrollTop;
      
      // Reset flag after a delay
      setTimeout(() => {
        isUserScrollingRef.current = false;
      }, 200);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      container.removeEventListener('scroll', handleScroll);
    };
  }, [containerRef]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || dependency === undefined) return;

    // Preserve scroll position when dependency changes
    if (!isUserScrollingRef.current) {
      requestAnimationFrame(() => {
        container.scrollTop = scrollPositionRef.current;
      });
    }
  }, [dependency, containerRef]);

  const preserveScroll = (callback: () => void) => {
    const container = containerRef.current;
    if (!container) {
      callback();
      return;
    }

    const currentScroll = container.scrollTop;
    callback();
    
    if (!isUserScrollingRef.current) {
      requestAnimationFrame(() => {
        container.scrollTop = currentScroll;
      });
    }
  };

  const scrollToTop = () => {
    const container = containerRef.current;
    if (container) {
      container.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
      scrollPositionRef.current = 0;
    }
  };

  const scrollToBottom = () => {
    const container = containerRef.current;
    if (container) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
      scrollPositionRef.current = container.scrollHeight;
    }
  };

  return {
    preserveScroll,
    scrollToTop,
    scrollToBottom,
    isUserScrolling: isUserScrollingRef.current
  };
};