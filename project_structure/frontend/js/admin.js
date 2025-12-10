// API Configuration
const API_BASE_URL = 'http://localhost:8001';
const API_V1 = `${API_BASE_URL}/api/v1`;

// Global state
let dashboardData = null;
let customersData = [];
let policiesData = [];

// Initialize admin dashboard
document.addEventListener('DOMContentLoaded', async () => {
    initializeNavigation();
    await loadDashboardData();
    initializeCharts();
    loadCustomersTable();
    loadPoliciesTable();
    loadActivityFeed();
    
    // Auto-refresh every 5 minutes
    setInterval(() => {
        refreshDashboard();
    }, 300000);
});

// Navigation
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.getAttribute('data-section');
            showSection(section);
            
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function showSection(sectionName) {
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(`section-${sectionName}`).classList.add('active');
}

// Load Dashboard Data
async function loadDashboardData() {
    try {
        // In production, fetch from API
        // For now, use simulated data based on database stats
        dashboardData = {
            totalCustomers: 191480,
            activePolicies: 52645,
            highRiskPolicies: 2847,
            totalPremium: 65789350,
            revenue: generateRevenueData(),
            policyDistribution: {
                'TPL': 22456,
                'COMP': 18234,
                'COLL': 11955
            },
            riskDistribution: {
                'Low': 38456,
                'Medium': 11342,
                'High': 2847
            }
        };

        updateKPIs();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

function updateKPIs() {
    animateValue('totalCustomers', dashboardData.totalCustomers);
    animateValue('activePolicies', dashboardData.activePolicies);
    animateValue('highRisk', dashboardData.highRiskPolicies);
    document.getElementById('totalPremium').textContent = 
        `$${(dashboardData.totalPremium / 1000000).toFixed(1)}M`;
}

function animateValue(id, target) {
    const element = document.getElementById(id);
    let current = 0;
    const increment = target / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current).toLocaleString();
    }, 20);
}

// Generate Revenue Data
function generateRevenueData() {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const data = [];
    let base = 4500000;
    
    for (let i = 0; i < 12; i++) {
        base += Math.random() * 500000 - 200000;
        data.push({
            month: months[i],
            revenue: Math.round(base)
        });
    }
    return data;
}

// Initialize Charts
function initializeCharts() {
    createRevenueChart();
    createPolicyDistChart();
    createRiskDistChart();
    createAgeDistChart();
    createVehicleTypeChart();
    createClaimsTrendChart();
    createFeatureImportanceChart();
    createModelPerformanceChart();
}

function createRevenueChart() {
    const ctx = document.getElementById('revenueChart');
    if (!ctx) return;

    const data = dashboardData.revenue;
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.month),
            datasets: [{
                label: 'Revenue',
                data: data.map(d => d.revenue),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000000).toFixed(1) + 'M';
                        }
                    }
                }
            }
        }
    });
}

function createPolicyDistChart() {
    const ctx = document.getElementById('policyDistChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Third Party Liability', 'Comprehensive', 'Collision'],
            datasets: [{
                data: [22456, 18234, 11955],
                backgroundColor: ['#667eea', '#10b981', '#f59e0b'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createRiskDistChart() {
    const ctx = document.getElementById('riskDistChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                label: 'Number of Policies',
                data: [38456, 11342, 2847],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function createAgeDistChart() {
    const ctx = document.getElementById('ageDistChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            datasets: [{
                label: 'Customers',
                data: [15234, 38456, 52345, 48567, 28456, 8422],
                backgroundColor: '#667eea',
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createVehicleTypeChart() {
    const ctx = document.getElementById('vehicleTypeChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Diesel', 'Petrol', 'Electric', 'Hybrid'],
            datasets: [{
                data: [28456, 18234, 3456, 2499],
                backgroundColor: ['#667eea', '#10b981', '#3b82f6', '#f59e0b']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createClaimsTrendChart() {
    const ctx = document.getElementById('claimsTrendChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            datasets: [{
                label: 'Claims Filed',
                data: [234, 256, 189, 298, 267, 245, 289, 312, 278, 245, 267, 289],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Claims Settled',
                data: [198, 223, 167, 278, 245, 223, 267, 289, 256, 234, 245, 267],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createFeatureImportanceChart() {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Premium', 'Claims History', 'Vehicle Age', 'Driver Age', 'Vehicle Value', 'Payment Type'],
            datasets: [{
                label: 'Importance',
                data: [0.28, 0.24, 0.18, 0.15, 0.10, 0.05],
                backgroundColor: '#667eea',
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 0.3
                }
            }
        }
    });
}

function createModelPerformanceChart() {
    const ctx = document.getElementById('modelPerformanceChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'],
            datasets: [{
                label: 'Accuracy',
                data: [0.9234, 0.9312, 0.9389, 0.9356, 0.9405, 0.9398, 0.9412, 0.9405],
                borderColor: '#10b981',
                tension: 0.4
            }, {
                label: 'Precision',
                data: [0.9156, 0.9234, 0.9267, 0.9289, 0.9312, 0.9278, 0.9289, 0.9261],
                borderColor: '#3b82f6',
                tension: 0.4
            }, {
                label: 'Recall',
                data: [0.9312, 0.9356, 0.9389, 0.9398, 0.9412, 0.9405, 0.9418, 0.9405],
                borderColor: '#f59e0b',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0.9,
                    max: 1.0
                }
            }
        }
    });
}

// Load Customers Table
function loadCustomersTable() {
    const tbody = document.getElementById('customersTableBody');
    if (!tbody) return;

    // Generate sample customer data
    const sampleCustomers = [
        {id: 'C001234', name: 'John Smith', age: 42, policies: 2, premium: 2450, risk: 0.23},
        {id: 'C001235', name: 'Emma Johnson', age: 35, policies: 1, premium: 1890, risk: 0.15},
        {id: 'C001236', name: 'Michael Brown', age: 28, policies: 1, premium: 3200, risk: 0.68},
        {id: 'C001237', name: 'Sarah Davis', age: 51, policies: 3, premium: 4560, risk: 0.42},
        {id: 'C001238', name: 'David Wilson', age: 39, policies: 2, premium: 2890, risk: 0.31},
        {id: 'C001239', name: 'Lisa Anderson', age: 45, policies: 1, premium: 2100, risk: 0.19},
        {id: 'C001240', name: 'James Martinez', age: 33, policies: 2, premium: 2750, risk: 0.27},
        {id: 'C001241', name: 'Jennifer Taylor', age: 48, policies: 1, premium: 1950, risk: 0.21}
    ];

    tbody.innerHTML = sampleCustomers.map(customer => {
        const riskClass = customer.risk < 0.3 ? 'low' : customer.risk < 0.6 ? 'medium' : 'high';
        const riskColor = customer.risk < 0.3 ? 'success' : customer.risk < 0.6 ? 'warning' : 'danger';
        
        return `
            <tr>
                <td><strong>${customer.id}</strong></td>
                <td>${customer.name}</td>
                <td>${customer.age}</td>
                <td>${customer.policies}</td>
                <td>$${customer.premium.toLocaleString()}</td>
                <td>
                    <span class="risk-badge ${riskClass}">
                        ${(customer.risk * 100).toFixed(0)}%
                    </span>
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="viewCustomer('${customer.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" onclick="editCustomer('${customer.id}')">
                        <i class="fas fa-edit"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Load Policies Table
function loadPoliciesTable() {
    const tbody = document.getElementById('policiesTableBody');
    if (!tbody) return;

    const samplePolicies = [
        {id: 'POL-12345', customer: 'John Smith', type: 'COMP', vehicle: '2018 Toyota Camry', premium: 1250, start: '2024-01-15', renewal: '2025-01-15', status: 'Active'},
        {id: 'POL-12346', customer: 'Emma Johnson', type: 'TPL', vehicle: '2020 Honda Civic', premium: 890, start: '2024-03-20', renewal: '2025-03-20', status: 'Active'},
        {id: 'POL-12347', customer: 'Michael Brown', type: 'COMP', vehicle: '2017 BMW X5', premium: 3200, start: '2024-02-10', renewal: '2025-02-10', status: 'Active'},
        {id: 'POL-12348', customer: 'Sarah Davis', type: 'COLL', vehicle: '2019 Mercedes C-Class', premium: 2100, start: '2024-05-05', renewal: '2024-12-20', status: 'Expiring'}
    ];

    tbody.innerHTML = samplePolicies.map(policy => {
        const statusClass = policy.status === 'Active' ? 'success' : 'warning';
        return `
            <tr>
                <td><strong>${policy.id}</strong></td>
                <td>${policy.customer}</td>
                <td><span class="badge bg-secondary">${policy.type}</span></td>
                <td>${policy.vehicle}</td>
                <td>$${policy.premium.toLocaleString()}</td>
                <td>${policy.start}</td>
                <td>${policy.renewal}</td>
                <td><span class="badge bg-${statusClass}">${policy.status}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Load Activity Feed
function loadActivityFeed() {
    const activityList = document.getElementById('activityList');
    if (!activityList) return;

    const activities = [
        {icon: 'fa-user-plus', color: 'bg-success', text: 'New customer registered', time: '2 minutes ago'},
        {icon: 'fa-file-contract', color: 'bg-primary', text: 'Policy POL-12350 renewed', time: '15 minutes ago'},
        {icon: 'fa-exclamation-triangle', color: 'bg-warning', text: 'High risk policy detected', time: '1 hour ago'},
        {icon: 'fa-check-circle', color: 'bg-success', text: 'Claim CLM-2025-042 approved', time: '2 hours ago'},
        {icon: 'fa-envelope', color: 'bg-info', text: 'Renewal reminder sent to 150 customers', time: '3 hours ago'}
    ];

    activityList.innerHTML = activities.map(activity => `
        <div class="activity-item">
            <div class="activity-icon ${activity.color} text-white">
                <i class="fas ${activity.icon}"></i>
            </div>
            <div class="activity-content">
                <h6>${activity.text}</h6>
                <p>${activity.time}</p>
            </div>
        </div>
    `).join('');
}

// Utility Functions
function refreshDashboard() {
    loadDashboardData();
    loadCustomersTable();
    loadPoliciesTable();
    loadActivityFeed();
}

function exportReport() {
    alert('Exporting comprehensive report...');
}

function exportRiskReport() {
    alert('Exporting high-risk policies report...');
}

function viewCustomer(id) {
    alert(`Viewing customer: ${id}`);
}

function editCustomer(id) {
    alert(`Editing customer: ${id}`);
}

// Sidebar Toggle for Mobile
document.getElementById('sidebarToggle')?.addEventListener('click', () => {
    document.querySelector('.sidebar').classList.toggle('active');
});

// Update prediction count
let predictionCount = 0;
setInterval(() => {
    predictionCount += Math.floor(Math.random() * 5);
    const elem = document.getElementById('predictionCount');
    if (elem) elem.textContent = predictionCount.toLocaleString();
}, 3000);

// Update policy counts
setTimeout(() => {
    document.getElementById('activePoliciesCount').textContent = '52,645';
    document.getElementById('expiringSoon').textContent = '2,847';
    document.getElementById('lapsedPolicies').textContent = '1,234';
}, 1000);

// Search functionality
document.getElementById('globalSearch')?.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    // Implement search logic here
});

document.getElementById('customerSearch')?.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    // Implement customer search logic here
});

console.log('✓ Admin dashboard initialized');
console.log(`✓ Loaded ${dashboardData?.totalCustomers.toLocaleString()} customer records`);
console.log(`✓ Tracking ${dashboardData?.activePolicies.toLocaleString()} active policies`);

// ==================== AI ASSISTANT FUNCTIONS ====================

let aiQueryCount = 0;
let aiRiskCount = 0;
let aiRecommendCount = 0;

// Main AI Query Function
async function askAI() {
    const queryInput = document.getElementById('aiQuery');
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a question');
        return;
    }
    
    // Add user message to chat
    addMessageToChat('user', query);
    queryInput.value = '';
    
    // Show loading message
    const loadingId = addMessageToChat('ai', 'Thinking<span class="loading-dots"></span>', true);
    
    try {
        const response = await fetch(`${API_V1}/llm/underwriter-assist`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) throw new Error('AI service unavailable');
        
        const data = await response.json();
        
        // Remove loading message and add real response
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', data.answer);
        
        // Update stats
        aiQueryCount++;
        updateAIStats();
        
        // Log interaction
        logAIInteraction('Query', query);
        
    } catch (error) {
        console.error('AI query failed:', error);
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', '⚠️ Sorry, I\'m having trouble connecting. Please try again.');
    }
}

// Quick Risk Assessment
async function quickRiskAssessment() {
    const policyId = document.getElementById('policyIdRisk').value;
    
    if (!policyId) {
        alert('Please enter a Policy ID');
        return;
    }
    
    addMessageToChat('user', `Assess risk for Policy #${policyId}`);
    const loadingId = addMessageToChat('ai', 'Analyzing risk<span class="loading-dots"></span>', true);
    
    try {
        // Mock policy data - in production, fetch from database
        const policyData = {
            policy_id: policyId,
            make_model: "2019 Toyota Camry",
            age: "6 years",
            power: "203 HP",
            fuel_type: "Petrol",
            value: "$22,000",
            claims_last_year: 0
        };
        
        const response = await fetch(`${API_V1}/llm/assess-risk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ policy_data: policyData })
        });
        
        if (!response.ok) throw new Error('Risk assessment failed');
        
        const data = await response.json();
        
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', `
            <strong>Risk Level: ${data.risk_level.toUpperCase()}</strong><br>
            ${data.assessment}
        `);
        
        aiRiskCount++;
        updateAIStats();
        logAIInteraction('Risk Assessment', `Policy ${policyId}`);
        
    } catch (error) {
        console.error('Risk assessment failed:', error);
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', '⚠️ Risk assessment unavailable. Check API connection.');
    }
}

// Explain Decision
async function explainDecision() {
    const policyId = document.getElementById('policyIdDecision').value;
    const decision = document.getElementById('decisionType').value;
    
    if (!policyId) {
        alert('Please enter a Policy ID');
        return;
    }
    
    addMessageToChat('user', `Explain ${decision} decision for Policy #${policyId}`);
    const loadingId = addMessageToChat('ai', 'Generating explanation<span class="loading-dots"></span>', true);
    
    try {
        const reason = decision === 'approved' ? 'approved based on good risk profile' : 
                      decision === 'denied' ? 'denied due to high risk factors' : 
                      'under review for additional information';
        
        const response = await fetch(`${API_V1}/llm/explain-decision?policy_id=${policyId}&decision=${decision}&reason=${reason}`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Explanation failed');
        
        const data = await response.json();
        
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', data.explanation);
        
        aiQueryCount++;
        updateAIStats();
        logAIInteraction('Decision Explanation', `Policy ${policyId} - ${decision}`);
        
    } catch (error) {
        console.error('Explanation failed:', error);
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', '⚠️ Unable to generate explanation. Please try again.');
    }
}

// Get Recommendation
async function getRecommendation() {
    const profileText = document.getElementById('customerProfile').value.trim();
    
    if (!profileText) {
        alert('Please enter customer information');
        return;
    }
    
    addMessageToChat('user', `Get policy recommendation for: ${profileText}`);
    const loadingId = addMessageToChat('ai', 'Analyzing profile<span class="loading-dots"></span>', true);
    
    try {
        const customerProfile = {
            description: profileText,
            age: 35,
            vehicle: "sedan",
            claims_history: "0"
        };
        
        const response = await fetch(`${API_V1}/llm/recommend-policy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ customer_profile: customerProfile })
        });
        
        if (!response.ok) throw new Error('Recommendation failed');
        
        const data = await response.json();
        
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', `
            <strong>Policy Recommendation:</strong><br>
            ${data.recommendation}
        `);
        
        aiRecommendCount++;
        updateAIStats();
        logAIInteraction('Recommendation', profileText.substring(0, 50));
        
    } catch (error) {
        console.error('Recommendation failed:', error);
        document.querySelector(`[data-message-id="${loadingId}"]`)?.remove();
        addMessageToChat('ai', '⚠️ Unable to generate recommendation. Please try again.');
    }
}

// Helper Functions
function addMessageToChat(type, content, isLoading = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = type === 'user' ? 'user-message' : 'ai-message';
    messageDiv.setAttribute('data-message-id', messageId);
    
    const avatar = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    const name = type === 'user' ? 'You' : 'AI Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <strong>${name}</strong>
            <p>${content}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

function updateAIStats() {
    document.getElementById('aiQueriesCount').textContent = aiQueryCount;
    document.getElementById('aiRiskCount').textContent = aiRiskCount;
    document.getElementById('aiRecommendCount').textContent = aiRecommendCount;
}

function logAIInteraction(type, query) {
    const table = document.getElementById('aiHistoryTable');
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Remove "no interactions" row if present
    if (table.rows.length === 1 && table.rows[0].cells.length === 1) {
        table.innerHTML = '';
    }
    
    const row = table.insertRow(0);
    row.innerHTML = `
        <td>${timeStr}</td>
        <td>Admin User</td>
        <td><span class="badge bg-primary">${type}</span></td>
        <td>${query.substring(0, 50)}${query.length > 50 ? '...' : ''}</td>
        <td><span class="badge bg-success">Success</span></td>
    `;
    
    // Keep only last 10 interactions
    while (table.rows.length > 10) {
        table.deleteRow(10);
    }
}
