import { tradeHistoryService } from './persistence/tradeHistoryService';
import { portfolioAnalytics } from './portfolioAnalytics';
import { virtualPortfolioService } from './persistence/virtualPortfolioService';

export interface ExportOptions {
  format: 'csv' | 'pdf';
  type: 'trades' | 'performance' | 'tax_report' | 'full_report';
  dateRange?: {
    start: string;
    end: string;
  };
  includeMetrics?: boolean;
}

export interface TradeExport {
  date: string;
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number | null;
  fees: number;
  strategy: string;
  confidence: number;
  status: string;
}

export interface PerformanceExport {
  date: string;
  portfolio_value: number;
  daily_return: number;
  cumulative_return: number;
  drawdown: number;
  trades_count: number;
  win_rate: number;
  sharpe_ratio: number;
}

class ExportService {
  /**
   * Export trade history as CSV
   */
  async exportTrades(options: ExportOptions): Promise<string> {
    try {
      const trades = await tradeHistoryService.getTradesByDateRange(
        options.dateRange?.start || this.getDefaultStartDate(),
        options.dateRange?.end || new Date().toISOString()
      );

      const exportData: TradeExport[] = trades.map(trade => ({
        date: new Date(trade.created_at).toLocaleDateString(),
        symbol: trade.symbol,
        side: trade.side,
        quantity: trade.quantity,
        entry_price: trade.entry_price,
        exit_price: trade.exit_price,
        pnl: trade.pnl,
        fees: trade.fees || 0,
        strategy: trade.strategy_attribution || 'Unknown',
        confidence: trade.confidence_score,
        status: trade.execution_status
      }));

      return this.convertToCSV(exportData, [
        'Date',
        'Symbol',
        'Side',
        'Quantity',
        'Entry Price',
        'Exit Price',
        'P&L',
        'Fees',
        'Strategy',
        'Confidence',
        'Status'
      ]);
    } catch (error) {
      console.error('Error exporting trades:', error);
      throw new Error('Failed to export trade data');
    }
  }

  /**
   * Export performance metrics as CSV
   */
  async exportPerformance(options: ExportOptions): Promise<string> {
    try {
      const snapshots = await virtualPortfolioService.getDailySnapshots(
        options.dateRange ? this.getDaysBetween(options.dateRange.start, options.dateRange.end) : 90
      );

      const metrics = await portfolioAnalytics.calculateMetrics(90);

      const exportData: PerformanceExport[] = snapshots.map((snapshot, index) => {
        const prevValue = index > 0 ? snapshots[index - 1].total_value : 50000;
        const dailyReturn = ((snapshot.total_value - prevValue) / prevValue) * 100;
        const cumulativeReturn = ((snapshot.total_value - 50000) / 50000) * 100;
        
        return {
          date: new Date(snapshot.snapshot_date).toLocaleDateString(),
          portfolio_value: snapshot.total_value,
          daily_return: dailyReturn,
          cumulative_return: cumulativeReturn,
          drawdown: 0, // Would be calculated based on peak
          trades_count: snapshot.trades_count || 0,
          win_rate: 0, // Would be calculated from recent trades
          sharpe_ratio: metrics.sharpeRatio
        };
      });

      if (options.includeMetrics) {
        // Add summary metrics at the top
        const metricsRows = [
          ['Metric', 'Value'],
          ['Total Return', `${metrics.totalReturnPercent.toFixed(2)}%`],
          ['Sharpe Ratio', metrics.sharpeRatio.toString()],
          ['Sortino Ratio', metrics.sortinoRatio.toString()],
          ['Max Drawdown', `${metrics.maxDrawdownPercent.toFixed(2)}%`],
          ['Win Rate', `${metrics.winRate.toFixed(2)}%`],
          ['Profit Factor', metrics.profitFactor.toString()],
          ['Total Trades', metrics.totalTrades.toString()],
          ['', ''], // Empty row separator
          ['Date', 'Portfolio Value', 'Daily Return %', 'Cumulative Return %', 'Drawdown %', 'Trades Count', 'Win Rate %', 'Sharpe Ratio']
        ];

        const dataRows = exportData.map(row => [
          row.date,
          row.portfolio_value.toString(),
          row.daily_return.toFixed(2),
          row.cumulative_return.toFixed(2),
          row.drawdown.toFixed(2),
          row.trades_count.toString(),
          row.win_rate.toFixed(2),
          row.sharpe_ratio.toString()
        ]);

        return [...metricsRows, ...dataRows].map(row => row.join(',')).join('\n');
      }

      return this.convertToCSV(exportData, [
        'Date',
        'Portfolio Value',
        'Daily Return %',
        'Cumulative Return %',
        'Drawdown %',
        'Trades Count',
        'Win Rate %',
        'Sharpe Ratio'
      ]);
    } catch (error) {
      console.error('Error exporting performance:', error);
      throw new Error('Failed to export performance data');
    }
  }

  /**
   * Generate tax report with P&L information
   */
  async exportTaxReport(options: ExportOptions): Promise<string> {
    try {
      const startOfYear = new Date(new Date().getFullYear(), 0, 1).toISOString();
      const endOfYear = new Date(new Date().getFullYear(), 11, 31).toISOString();

      const trades = await tradeHistoryService.getTradesByDateRange(
        options.dateRange?.start || startOfYear,
        options.dateRange?.end || endOfYear
      );

      const completedTrades = trades.filter(trade => trade.exit_price && trade.pnl !== null);

      const taxData = completedTrades.map(trade => {
        const holdingPeriod = trade.exit_time && trade.entry_time 
          ? Math.ceil((new Date(trade.exit_time).getTime() - new Date(trade.entry_time).getTime()) / (1000 * 60 * 60 * 24))
          : 0;

        const isLongTerm = holdingPeriod > 365;

        return {
          date_acquired: new Date(trade.entry_time || trade.created_at).toLocaleDateString(),
          date_sold: new Date(trade.exit_time || '').toLocaleDateString(),
          symbol: trade.symbol,
          quantity: trade.quantity,
          cost_basis: (trade.entry_price * trade.quantity).toFixed(2),
          sale_proceeds: trade.exit_price ? (trade.exit_price * trade.quantity).toFixed(2) : '0',
          gain_loss: trade.pnl?.toFixed(2) || '0',
          term: isLongTerm ? 'Long-term' : 'Short-term',
          holding_period_days: holdingPeriod
        };
      });

      // Calculate totals
      const shortTermGains = taxData.filter(t => t.term === 'Short-term').reduce((sum, t) => sum + parseFloat(t.gain_loss), 0);
      const longTermGains = taxData.filter(t => t.term === 'Long-term').reduce((sum, t) => sum + parseFloat(t.gain_loss), 0);
      const totalGains = shortTermGains + longTermGains;

      // Add summary at the top
      const summaryRows = [
        ['TAX REPORT SUMMARY'],
        ['Report Period', `${options.dateRange?.start || startOfYear} to ${options.dateRange?.end || endOfYear}`],
        ['Total Trades', completedTrades.length.toString()],
        ['Short-term Gains/Losses', `$${shortTermGains.toFixed(2)}`],
        ['Long-term Gains/Losses', `$${longTermGains.toFixed(2)}`],
        ['Total Gains/Losses', `$${totalGains.toFixed(2)}`],
        [''],
        ['INDIVIDUAL TRADES'],
        ['Date Acquired', 'Date Sold', 'Symbol', 'Quantity', 'Cost Basis', 'Sale Proceeds', 'Gain/Loss', 'Term', 'Holding Period (Days)']
      ];

      const dataRows = taxData.map(row => [
        row.date_acquired,
        row.date_sold,
        row.symbol,
        row.quantity.toString(),
        row.cost_basis,
        row.sale_proceeds,
        row.gain_loss,
        row.term,
        row.holding_period_days.toString()
      ]);

      return [...summaryRows, ...dataRows].map(row => row.join(',')).join('\n');
    } catch (error) {
      console.error('Error generating tax report:', error);
      throw new Error('Failed to generate tax report');
    }
  }

  /**
   * Generate comprehensive PDF report (HTML for now, can be converted to PDF)
   */
  async generatePDFReport(options: ExportOptions): Promise<string> {
    try {
      const metrics = await portfolioAnalytics.calculateMetrics(90);
      const recentTrades = await tradeHistoryService.getRecentTrades(20);
      const portfolio = await virtualPortfolioService.getPortfolio();

      const html = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>Trading Performance Report</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
            .header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #333; padding-bottom: 20px; }
            .section { margin: 30px 0; }
            .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
            .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2563eb; }
            .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
            .trades-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .trades-table th, .trades-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            .trades-table th { background-color: #f5f5f5; }
            .positive { color: #059669; }
            .negative { color: #dc2626; }
            .footer { margin-top: 40px; text-align: center; font-size: 12px; color: #666; border-top: 1px solid #ddd; padding-top: 20px; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Trading Performance Report</h1>
            <p>Generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}</p>
            <p>Portfolio Value: $${metrics.totalValue.toLocaleString()}</p>
          </div>

          <div class="section">
            <h2>Key Performance Metrics</h2>
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-value ${metrics.totalReturnPercent >= 0 ? 'positive' : 'negative'}">
                  ${metrics.totalReturnPercent.toFixed(2)}%
                </div>
                <div class="metric-label">Total Return</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.sharpeRatio.toFixed(2)}</div>
                <div class="metric-label">Sharpe Ratio</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.winRate.toFixed(1)}%</div>
                <div class="metric-label">Win Rate</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.maxDrawdownPercent.toFixed(2)}%</div>
                <div class="metric-label">Max Drawdown</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.profitFactor.toFixed(2)}</div>
                <div class="metric-label">Profit Factor</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.totalTrades}</div>
                <div class="metric-label">Total Trades</div>
              </div>
            </div>
          </div>

          <div class="section">
            <h2>Risk Metrics</h2>
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-value">${metrics.volatility.toFixed(2)}%</div>
                <div class="metric-label">Volatility</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.sortinoRatio.toFixed(2)}</div>
                <div class="metric-label">Sortino Ratio</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">${metrics.calmarRatio.toFixed(2)}</div>
                <div class="metric-label">Calmar Ratio</div>
              </div>
            </div>
          </div>

          <div class="section">
            <h2>Recent Trades (Last 20)</h2>
            <table class="trades-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Quantity</th>
                  <th>Entry Price</th>
                  <th>Exit Price</th>
                  <th>P&L</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                ${recentTrades.map(trade => `
                  <tr>
                    <td>${new Date(trade.created_at).toLocaleDateString()}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.side.toUpperCase()}</td>
                    <td>${trade.quantity}</td>
                    <td>$${trade.entry_price.toFixed(2)}</td>
                    <td>${trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}</td>
                    <td class="${(trade.pnl || 0) >= 0 ? 'positive' : 'negative'}">
                      ${trade.pnl ? `$${trade.pnl.toFixed(2)}` : '-'}
                    </td>
                    <td>${trade.execution_status}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>

          <div class="footer">
            <p>This report was generated automatically by the Trading Bot system.</p>
            <p>For questions or support, please contact your trading platform administrator.</p>
          </div>
        </body>
        </html>
      `;

      return html;
    } catch (error) {
      console.error('Error generating PDF report:', error);
      throw new Error('Failed to generate PDF report');
    }
  }

  /**
   * Download exported data as file
   */
  downloadFile(data: string, filename: string, type: 'csv' | 'html' = 'csv'): void {
    const mimeType = type === 'csv' ? 'text/csv' : 'text/html';
    const blob = new Blob([data], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.${type}`;
    link.click();
    
    window.URL.revokeObjectURL(url);
  }

  /**
   * Helper: Convert data array to CSV format
   */
  private convertToCSV(data: any[], headers: string[]): string {
    const csvHeaders = headers.join(',');
    const csvRows = data.map(row => {
      return headers.map(header => {
        const key = header.toLowerCase().replace(/[^a-z0-9]/g, '_');
        const value = row[key] !== undefined ? row[key] : '';
        return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
      }).join(',');
    });

    return [csvHeaders, ...csvRows].join('\n');
  }

  /**
   * Helper: Get default start date (30 days ago)
   */
  private getDefaultStartDate(): string {
    const date = new Date();
    date.setDate(date.getDate() - 30);
    return date.toISOString();
  }

  /**
   * Helper: Calculate days between two dates
   */
  private getDaysBetween(start: string, end: string): number {
    const startDate = new Date(start);
    const endDate = new Date(end);
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }
}

export const exportService = new ExportService();
export default exportService;