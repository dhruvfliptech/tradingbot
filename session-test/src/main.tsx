import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import AppRouter from './AppRouter.tsx';
import './index.css';

console.log('Main.tsx loading...');

try {
  const rootElement = document.getElementById('root');
  if (!rootElement) {
    console.error('Root element not found!');
    document.body.innerHTML = '<div style="color: red; padding: 20px;">Root element not found!</div>';
  } else {
    console.log('Creating React root...');
    createRoot(rootElement).render(
      <StrictMode>
        <AppRouter />
      </StrictMode>
    );
    console.log('React app rendered');
  }
} catch (error) {
  console.error('Error rendering app:', error);
  document.body.innerHTML = `<div style="color: red; padding: 20px;">Error: ${error}</div>`;
}
