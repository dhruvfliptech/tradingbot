import React, { useState } from 'react';
import { 
  Download, 
  FileText, 
  BarChart3, 
  Calculator, 
  Calendar,
  Settings,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { exportService, ExportOptions } from '../../services/exportService';

type ExportType = 'trades' | 'performance' | 'tax_report' | 'full_report';
type ExportFormat = 'csv' | 'pdf';

const ExportReporting: React.FC = () => {
  const [selectedType, setSelectedType] = useState<ExportType>('trades');
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('csv');
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 30 days ago
    end: new Date().toISOString().split('T')[0] // today
  });
  const [includeMetrics, setIncludeMetrics] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState<{ type: 'success' | 'error', message: string } | null>(null);

  const exportTypes = [
    {
      id: 'trades' as ExportType,
      name: 'Trade History',
      description: 'Export all trading activity with entry/exit prices, P&L, and strategy attribution',
      icon: BarChart3,
      formats: ['csv'] as ExportFormat[]
    },
    {
      id: 'performance' as ExportType,
      name: 'Performance Metrics',
      description: 'Daily portfolio values, returns, and risk metrics',
      icon: FileText,
      formats: ['csv'] as ExportFormat[]
    },
    {
      id: 'tax_report' as ExportType,
      name: 'Tax Report',
      description: 'Capital gains/losses report for tax filing purposes',
      icon: Calculator,
      formats: ['csv'] as ExportFormat[]
    },
    {
      id: 'full_report' as ExportType,
      name: 'Full Report',
      description: 'Comprehensive performance report with charts and analysis',
      icon: FileText,
      formats: ['pdf'] as ExportFormat[]
    }
  ];

  const handleExport = async () => {
    setIsExporting(true);
    setExportStatus(null);

    try {
      const options: ExportOptions = {
        format: selectedFormat,
        type: selectedType,
        dateRange: {
          start: new Date(dateRange.start).toISOString(),
          end: new Date(dateRange.end).toISOString()
        },
        includeMetrics: includeMetrics
      };

      let exportData: string;
      let filename: string;
      let fileType: 'csv' | 'html' = 'csv';

      switch (selectedType) {
        case 'trades':
          exportData = await exportService.exportTrades(options);
          filename = `trades_${dateRange.start}_to_${dateRange.end}`;
          break;
        case 'performance':
          exportData = await exportService.exportPerformance(options);
          filename = `performance_${dateRange.start}_to_${dateRange.end}`;
          break;
        case 'tax_report':
          exportData = await exportService.exportTaxReport(options);
          filename = `tax_report_${new Date().getFullYear()}`;
          break;
        case 'full_report':
          exportData = await exportService.generatePDFReport(options);
          filename = `full_report_${new Date().toISOString().split('T')[0]}`;
          fileType = 'html';
          break;
        default:
          throw new Error('Invalid export type');
      }

      // Download the file
      exportService.downloadFile(exportData, filename, fileType);

      setExportStatus({
        type: 'success',
        message: `${exportTypes.find(t => t.id === selectedType)?.name} exported successfully!`
      });

    } catch (error) {
      console.error('Export error:', error);
      setExportStatus({
        type: 'error',
        message: `Export failed: ${error.message}`
      });
    } finally {
      setIsExporting(false);
    }
  };

  const selectedExportType = exportTypes.find(t => t.id === selectedType);

  return (
    <div className="bg-gray-900 rounded-lg p-6 space-y-6">
      <div className="flex items-center space-x-3">
        <Download className="w-6 h-6 text-blue-400" />
        <h3 className="text-xl font-semibold text-white">Export & Reporting</h3>
      </div>

      {/* Export Type Selection */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium text-white">Select Report Type</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {exportTypes.map((type) => {
            const Icon = type.icon;
            const isSelected = selectedType === type.id;
            
            return (
              <div
                key={type.id}
                onClick={() => setSelectedType(type.id)}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  isSelected 
                    ? 'border-blue-500 bg-blue-900/20' 
                    : 'border-gray-700 bg-gray-800 hover:border-gray-600'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <Icon className={`w-6 h-6 mt-1 ${isSelected ? 'text-blue-400' : 'text-gray-400'}`} />
                  <div className="flex-1">
                    <h5 className={`font-medium ${isSelected ? 'text-blue-300' : 'text-white'}`}>
                      {type.name}
                    </h5>
                    <p className="text-sm text-gray-400 mt-1">{type.description}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Format Selection */}
      <div className="space-y-3">
        <h4 className="text-lg font-medium text-white">Export Format</h4>
        <div className="flex space-x-4">
          {selectedExportType?.formats.map((format) => (
            <label key={format} className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="format"
                value={format}
                checked={selectedFormat === format}
                onChange={(e) => setSelectedFormat(e.target.value as ExportFormat)}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-white uppercase">{format}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Date Range Selection */}
      {selectedType !== 'full_report' && (
        <div className="space-y-3">
          <h4 className="text-lg font-medium text-white flex items-center">
            <Calendar className="w-5 h-5 mr-2" />
            Date Range
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Start Date</label>
              <input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                className="w-full bg-gray-800 text-white px-3 py-2 rounded border border-gray-700 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">End Date</label>
              <input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                className="w-full bg-gray-800 text-white px-3 py-2 rounded border border-gray-700 focus:border-blue-500"
              />
            </div>
          </div>
        </div>
      )}

      {/* Additional Options */}
      {selectedType === 'performance' && (
        <div className="space-y-3">
          <h4 className="text-lg font-medium text-white flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Options
          </h4>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={includeMetrics}
              onChange={(e) => setIncludeMetrics(e.target.checked)}
              className="text-blue-600 focus:ring-blue-500"
            />
            <span className="text-white">Include performance metrics summary</span>
          </label>
        </div>
      )}

      {/* Quick Export Presets */}
      <div className="space-y-3">
        <h4 className="text-lg font-medium text-white">Quick Exports</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            onClick={() => {
              setSelectedType('trades');
              setDateRange({
                start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                end: new Date().toISOString().split('T')[0]
              });
              setTimeout(handleExport, 100);
            }}
            disabled={isExporting}
            className="bg-gray-800 hover:bg-gray-700 px-4 py-3 rounded-lg text-white text-sm transition-colors border border-gray-700 hover:border-gray-600"
          >
            <div className="text-left">
              <div className="font-medium">Last 7 Days</div>
              <div className="text-xs text-gray-400">Trade History</div>
            </div>
          </button>
          
          <button
            onClick={() => {
              setSelectedType('performance');
              setDateRange({
                start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                end: new Date().toISOString().split('T')[0]
              });
              setTimeout(handleExport, 100);
            }}
            disabled={isExporting}
            className="bg-gray-800 hover:bg-gray-700 px-4 py-3 rounded-lg text-white text-sm transition-colors border border-gray-700 hover:border-gray-600"
          >
            <div className="text-left">
              <div className="font-medium">Last Month</div>
              <div className="text-xs text-gray-400">Performance</div>
            </div>
          </button>
          
          <button
            onClick={() => {
              setSelectedType('tax_report');
              setDateRange({
                start: `${new Date().getFullYear()}-01-01`,
                end: new Date().toISOString().split('T')[0]
              });
              setTimeout(handleExport, 100);
            }}
            disabled={isExporting}
            className="bg-gray-800 hover:bg-gray-700 px-4 py-3 rounded-lg text-white text-sm transition-colors border border-gray-700 hover:border-gray-600"
          >
            <div className="text-left">
              <div className="font-medium">Year to Date</div>
              <div className="text-xs text-gray-400">Tax Report</div>
            </div>
          </button>
        </div>
      </div>

      {/* Status Message */}
      {exportStatus && (
        <div className={`p-4 rounded-lg flex items-center space-x-3 ${
          exportStatus.type === 'success' 
            ? 'bg-green-900/20 border border-green-700' 
            : 'bg-red-900/20 border border-red-700'
        }`}>
          {exportStatus.type === 'success' ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <AlertCircle className="w-5 h-5 text-red-400" />
          )}
          <span className={exportStatus.type === 'success' ? 'text-green-300' : 'text-red-300'}>
            {exportStatus.message}
          </span>
        </div>
      )}

      {/* Export Button */}
      <div className="pt-4 border-t border-gray-800">
        <button
          onClick={handleExport}
          disabled={isExporting}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-6 py-3 rounded-lg text-white font-medium transition-colors flex items-center justify-center space-x-2"
        >
          {isExporting ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Exporting...</span>
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              <span>
                Export {selectedExportType?.name} as {selectedFormat.toUpperCase()}
              </span>
            </>
          )}
        </button>
      </div>

      {/* Info Box */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h5 className="text-white font-medium mb-2">Export Information</h5>
        <ul className="text-sm text-gray-400 space-y-1">
          <li>• Exports are generated in real-time from your latest data</li>
          <li>• CSV files can be opened in Excel, Google Sheets, or any spreadsheet application</li>
          <li>• PDF reports include charts and visual analysis (HTML format for now)</li>
          <li>• Tax reports include holding period calculations for capital gains</li>
          <li>• All exported data is temporary and not stored on our servers</li>
        </ul>
      </div>
    </div>
  );
};

export { ExportReporting };
export default ExportReporting;