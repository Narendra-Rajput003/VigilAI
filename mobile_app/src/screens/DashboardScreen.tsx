import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Dimensions,
} from 'react-native';
import { LineChart, BarChart, PieChart } from 'react-native-chart-kit';
import { useNavigation } from '@react-navigation/native';
import { useAuth } from '../hooks/useAuth';
import { useRealTimeData } from '../hooks/useRealTimeData';
import { useEmergency } from '../hooks/useEmergency';
import { useAnalytics } from '../hooks/useAnalytics';

const { width } = Dimensions.get('window');

interface DashboardScreenProps {
  navigation: any;
}

const DashboardScreen: React.FC<DashboardScreenProps> = ({ navigation }) => {
  const { user } = useAuth();
  const { realTimeData, isConnected } = useRealTimeData();
  const { triggerEmergency } = useEmergency();
  const { analytics, loading } = useAnalytics();
  
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [refreshing, setRefreshing] = useState(false);

  const handleEmergency = () => {
    Alert.alert(
      'Emergency Alert',
      'Are you sure you want to trigger an emergency alert?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Confirm', 
          style: 'destructive',
          onPress: () => triggerEmergency()
        }
      ]
    );
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    // Refresh data logic
    setTimeout(() => setRefreshing(false), 1000);
  };

  const chartConfig = {
    backgroundColor: '#1e1e1e',
    backgroundGradientFrom: '#1e1e1e',
    backgroundGradientTo: '#2d2d2d',
    decimalPlaces: 1,
    color: (opacity = 1) => `rgba(26, 255, 146, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
    style: {
      borderRadius: 16,
    },
    propsForDots: {
      r: '6',
      strokeWidth: '2',
      stroke: '#ffa726',
    },
  };

  const fatigueData = {
    labels: ['6h', '12h', '18h', '24h'],
    datasets: [
      {
        data: [0.2, 0.4, 0.6, 0.3],
        color: (opacity = 1) => `rgba(255, 107, 107, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  const stressData = [
    { name: 'Low', population: 65, color: '#4CAF50', legendFontColor: '#7F7F7F' },
    { name: 'Medium', population: 25, color: '#FF9800', legendFontColor: '#7F7F7F' },
    { name: 'High', population: 10, color: '#F44336', legendFontColor: '#7F7F7F' },
  ];

  return (
    <ScrollView style={styles.container} refreshControl={
      <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
    }>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.greeting}>Hello, {user?.name || 'Driver'}</Text>
        <View style={styles.statusContainer}>
          <View style={[styles.statusDot, { backgroundColor: isConnected ? '#4CAF50' : '#F44336' }]} />
          <Text style={styles.statusText}>
            {isConnected ? 'Connected' : 'Offline'}
          </Text>
        </View>
      </View>

      {/* Real-time Status */}
      <View style={styles.statusCard}>
        <Text style={styles.cardTitle}>Current Status</Text>
        <View style={styles.statusGrid}>
          <View style={styles.statusItem}>
            <Text style={styles.statusLabel}>Fatigue Level</Text>
            <Text style={[styles.statusValue, { color: getFatigueColor(realTimeData.fatigueLevel) }]}>
              {realTimeData.fatigueLevel?.toFixed(1) || '0.0'}
            </Text>
          </View>
          <View style={styles.statusItem}>
            <Text style={styles.statusLabel}>Stress Level</Text>
            <Text style={[styles.statusValue, { color: getStressColor(realTimeData.stressLevel) }]}>
              {realTimeData.stressLevel?.toFixed(1) || '0.0'}
            </Text>
          </View>
          <View style={styles.statusItem}>
            <Text style={styles.statusLabel}>Confidence</Text>
            <Text style={styles.statusValue}>
              {(realTimeData.confidence * 100)?.toFixed(0) || '0'}%
            </Text>
          </View>
        </View>
      </View>

      {/* Emergency Button */}
      <TouchableOpacity style={styles.emergencyButton} onPress={handleEmergency}>
        <Text style={styles.emergencyButtonText}>🚨 EMERGENCY</Text>
      </TouchableOpacity>

      {/* Charts Section */}
      <View style={styles.chartsSection}>
        <Text style={styles.sectionTitle}>Analytics</Text>
        
        {/* Fatigue Trend Chart */}
        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>Fatigue Trend (24h)</Text>
          <LineChart
            data={fatigueData}
            width={width - 40}
            height={200}
            chartConfig={chartConfig}
            bezier
            style={styles.chart}
          />
        </View>

        {/* Stress Distribution Chart */}
        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>Stress Distribution</Text>
          <PieChart
            data={stressData}
            width={width - 40}
            height={200}
            chartConfig={chartConfig}
            accessor="population"
            backgroundColor="transparent"
            paddingLeft="15"
            style={styles.chart}
          />
        </View>
      </View>

      {/* Quick Actions */}
      <View style={styles.quickActions}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.actionGrid}>
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={() => navigation.navigate('Analytics')}
          >
            <Text style={styles.actionIcon}>📊</Text>
            <Text style={styles.actionText}>Analytics</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={() => navigation.navigate('Settings')}
          >
            <Text style={styles.actionIcon}>⚙️</Text>
            <Text style={styles.actionText}>Settings</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={() => navigation.navigate('Fleet')}
          >
            <Text style={styles.actionIcon}>🚗</Text>
            <Text style={styles.actionText}>Fleet</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={styles.actionButton}
            onPress={() => navigation.navigate('Profile')}
          >
            <Text style={styles.actionIcon}>👤</Text>
            <Text style={styles.actionText}>Profile</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Recent Alerts */}
      <View style={styles.alertsSection}>
        <Text style={styles.sectionTitle}>Recent Alerts</Text>
        {analytics.recentAlerts?.map((alert, index) => (
          <View key={index} style={styles.alertItem}>
            <View style={[styles.alertIcon, { backgroundColor: getAlertColor(alert.severity) }]}>
              <Text style={styles.alertIconText}>
                {getAlertIcon(alert.type)}
              </Text>
            </View>
            <View style={styles.alertContent}>
              <Text style={styles.alertTitle}>{alert.title}</Text>
              <Text style={styles.alertTime}>{alert.timestamp}</Text>
            </View>
            <View style={[styles.alertStatus, { backgroundColor: alert.resolved ? '#4CAF50' : '#FF9800' }]}>
              <Text style={styles.alertStatusText}>
                {alert.resolved ? 'Resolved' : 'Active'}
              </Text>
            </View>
          </View>
        ))}
      </View>
    </ScrollView>
  );
};

const getFatigueColor = (level: number) => {
  if (level < 0.3) return '#4CAF50';
  if (level < 0.7) return '#FF9800';
  return '#F44336';
};

const getStressColor = (level: number) => {
  if (level < 0.3) return '#4CAF50';
  if (level < 0.7) return '#FF9800';
  return '#F44336';
};

const getAlertColor = (severity: string) => {
  switch (severity) {
    case 'high': return '#F44336';
    case 'medium': return '#FF9800';
    case 'low': return '#4CAF50';
    default: return '#9E9E9E';
  }
};

const getAlertIcon = (type: string) => {
  switch (type) {
    case 'fatigue': return '😴';
    case 'stress': return '😰';
    case 'emergency': return '🚨';
    case 'system': return '⚠️';
    default: return 'ℹ️';
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1e1e1e',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 50,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    color: '#ffffff',
    fontSize: 14,
  },
  statusCard: {
    backgroundColor: '#2d2d2d',
    margin: 20,
    padding: 20,
    borderRadius: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 16,
  },
  statusGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statusItem: {
    alignItems: 'center',
  },
  statusLabel: {
    fontSize: 12,
    color: '#9E9E9E',
    marginBottom: 4,
  },
  statusValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  emergencyButton: {
    backgroundColor: '#F44336',
    margin: 20,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  emergencyButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  chartsSection: {
    margin: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 16,
  },
  chartContainer: {
    backgroundColor: '#2d2d2d',
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  chart: {
    borderRadius: 16,
  },
  quickActions: {
    margin: 20,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionButton: {
    backgroundColor: '#2d2d2d',
    width: (width - 60) / 2,
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  actionIcon: {
    fontSize: 24,
    marginBottom: 8,
  },
  actionText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  alertsSection: {
    margin: 20,
    marginBottom: 40,
  },
  alertItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2d2d2d',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  alertIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  alertIconText: {
    fontSize: 20,
  },
  alertContent: {
    flex: 1,
  },
  alertTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  alertTime: {
    color: '#9E9E9E',
    fontSize: 12,
  },
  alertStatus: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  alertStatusText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: 'bold',
  },
});

export default DashboardScreen;
