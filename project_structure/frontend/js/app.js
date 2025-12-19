// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_V1 = `${API_BASE_URL}/api/v1`;

// Animate counter numbers
function animateCounter(element, target) {
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

// Load statistics on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Fetch database statistics
        const response = await fetch(`${API_BASE_URL}/api/v1/stats/dashboard`); // Corrected endpoint
        if (response.ok) {
            const data = await response.json();
            // Animate counters if data available
            if (data.customers) animateCounter(document.getElementById('customerCount'), data.customers);
            if (data.policies) animateCounter(document.getElementById('policyCount'), data.policies);
        } else {
            // Fallback
            animateCounter(document.getElementById('customerCount'), 191480);
            animateCounter(document.getElementById('policyCount'), 52645);
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
        // Show default animated numbers
        animateCounter(document.getElementById('customerCount'), 150000);
        animateCounter(document.getElementById('policyCount'), 45000);
    }

    // Initialize form handlers
    initializeQuoteForm();
    initializeRenewalForm();
    initializeClaimsForm();
    initializeGenericChat(); // New Chat
});

// Multi-step Quote Form Logic
function initializeQuoteForm() {
    const form = document.getElementById('quoteForm');
    const steps = document.querySelectorAll('.form-step');
    const progressSteps = document.querySelectorAll('.progress-step');
    let currentStep = 1;

    // Next button handler
    document.querySelectorAll('.btn-next').forEach(btn => {
        btn.addEventListener('click', () => {
            if (validateStep(currentStep)) {
                currentStep++;
                showStep(currentStep);
            }
        });
    });

    // Previous button handler
    document.querySelectorAll('.btn-prev').forEach(btn => {
        btn.addEventListener('click', () => {
            currentStep--;
            showStep(currentStep);
        });
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!validateStep(currentStep)) return;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Show loading state
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="loading"></span> Calculating with AI...';
        submitBtn.disabled = true;

        try {
            // Calculate quote using ML model API
            const quote = await calculateQuote(data);
            displayQuoteResults(quote);

            // Scroll to results
            document.getElementById('quoteResults').scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            showNotification('Failed to calculate quote. Please try again.', 'error');
            console.error('Quote calculation error:', error);
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });

    function showStep(step) {
        steps.forEach((s, index) => {
            s.classList.toggle('active', index + 1 === step);
        });
        progressSteps.forEach((s, index) => {
            s.classList.toggle('active', index + 1 === step);
            s.classList.toggle('completed', index + 1 < step);
        });
    }

    function validateStep(step) {
        const currentStepElement = document.querySelector(`.form-step[data-step="${step}"]`);
        const inputs = currentStepElement.querySelectorAll('input[required], select[required]');
        let valid = true;

        inputs.forEach(input => {
            if (!input.checkValidity()) {
                input.reportValidity();
                valid = false;
            }
        });

        return valid;
    }
}

// Calculate quote directly from API
async function calculateQuote(formData) {
    // Construct request body matching QuoteRequest Pydantic model
    const payload = {
        cover_type: formData.type_risk || "comprehensive", // mapped
        vehicle_use: "private", // default or add field to form
        vehicle_value: parseFloat(formData.value_vehicle),
        vehicle_year: parseInt(formData.year_matriculation || 2020),
        engine_cc: parseInt(formData.cylinder_capacity || 1500),
        driver_age: calculateAge(formData.date_birth),
        driving_experience: calculateAge(formData.date_driving_licence), // rough estimate
        previous_claims: parseInt(formData.n_claims_history || 0),
        vehicle_make: formData.vehicle_make,
        vehicle_model: formData.vehicle_model,
        fuel_type: mapFuelType(formData.type_fuel),

        // Additional ML fields
        n_claims_history: parseInt(formData.n_claims_history || 0),
        n_claims_year: parseInt(formData.n_claims_year || 0),
        power: parseInt(formData.power || 100),
        weight: parseInt(formData.weight || 1000)
    };

    try {
        const response = await fetch(`${API_V1}/quote`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const result = await response.json();

        // Transform API response to match UI needs
        // API returns multiple quotes/insurers, we take the best one or display generic
        const bestQuote = result.quotes[0]; // Cheapest

        return {
            premium: bestQuote.premium,
            lapseRisk: result.churn_probability || 0.1,
            riskLevel: result.risk_level.toLowerCase(),
            recommendation: generateRecommendation(result.risk_level, result.churn_probability),
            formData: formData,
            mlConfidence: result.ml_confidence,
            insurerName: bestQuote.insurer_name
        };

    } catch (error) {
        console.error("API Call Failed", error);
        throw error;
    }
}

// Helpers
function calculateAge(dateString) {
    if (!dateString) return 30; // default
    const today = new Date();
    const birthDate = new Date(dateString);
    let age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
        age--;
    }
    return age;
}

function mapFuelType(code) {
    const map = { 'P': 'petrol', 'D': 'diesel', 'E': 'electric', 'H': 'hybrid' };
    return map[code] || 'petrol';
}

function generateRecommendation(riskLevel, churnProb) {
    if (riskLevel === 'Low' || riskLevel === 'low') {
        return 'Excellent! You qualify for our premium rates with low risk. We recommend proceeding with this policy.';
    } else if (riskLevel === 'Medium' || riskLevel === 'medium') {
        return 'Good profile with moderate risk. Consider adding additional coverage for better long-term protection.';
    } else {
        return 'Higher risk profile detected. We have adjusted the premium accordingly. Call us for a review.';
    }
}

// Display quote results
function displayQuoteResults(quote) {
    const resultsDiv = document.getElementById('quoteResults');
    const formContainer = document.querySelector('.quote-form-container form');

    // Hide form, show results
    formContainer.style.display = 'none';
    resultsDiv.style.display = 'block';

    // Update premium
    document.getElementById('estimatedPremium').textContent = `$${quote.premium.toLocaleString()}`;

    // Update risk gauge
    const gaugeFill = document.querySelector('.gauge-fill');
    const riskLabel = document.getElementById('riskLabel');

    gaugeFill.className = `gauge-fill ${quote.riskLevel}`;

    let riskText = '';
    let riskColor = '';
    if (quote.riskLevel === 'low') {
        riskText = 'Low Risk - Excellent';
        riskColor = 'text-success';
    } else if (quote.riskLevel === 'medium') {
        riskText = 'Medium Risk - Good';
        riskColor = 'text-warning';
    } else {
        riskText = 'High Risk - Review Needed';
        riskColor = 'text-danger';
    }

    riskLabel.innerHTML = `<span class="${riskColor}">${riskText}</span><br><small>${(quote.lapseRisk * 100).toFixed(1)}% lapse probability</small>`;

    // Update recommendation
    document.getElementById('recommendationText').textContent = quote.recommendation;

    showNotification('Quote calculated successfully!', 'success');
}

// Reset form
function resetForm() {
    document.getElementById('quoteForm').reset();
    document.querySelector('.quote-form-container form').style.display = 'block';
    document.getElementById('quoteResults').style.display = 'none';

    // Reset to step 1
    document.querySelectorAll('.form-step').forEach((step, index) => {
        step.classList.toggle('active', index === 0);
    });
    document.querySelectorAll('.progress-step').forEach((step, index) => {
        step.classList.toggle('active', index === 0);
        step.classList.remove('completed');
    });
}

// Apply for policy
function applyForPolicy() {
    showNotification('Policy application submitted! Our agent will contact you within 24 hours.', 'success');
    setTimeout(() => {
        resetForm();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 2000);
}

// Renewal form handler
function initializeRenewalForm() {
    const form = document.getElementById('renewalForm');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="loading"></span> Checking...';
        submitBtn.disabled = true;

        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1500));

            const resultsDiv = document.getElementById('renewalResults');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = `
                <div class="alert alert-success">
                    <h5><i class="fas fa-check-circle"></i> Renewal Available</h5>
                    <p>Policy #${data.policy_number} is eligible for renewal.</p>
                    <hr>
                    <p><strong>Current Premium:</strong> $1,250/year</p>
                    <p><strong>Renewal Premium:</strong> $1,180/year (5% loyalty discount)</p>
                    <p><strong>Renewal Date:</strong> ${new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toLocaleDateString()}</p>
                    <button class="btn btn-success mt-3" onclick="processRenewal()">
                        <i class="fas fa-check"></i> Renew Now
                    </button>
                </div>
            `;

            showNotification('Renewal information retrieved successfully!', 'success');
        } catch (error) {
            showNotification('Failed to retrieve renewal information.', 'error');
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });
}

function processRenewal() {
    showNotification('Policy renewed successfully! Confirmation email sent.', 'success');
    setTimeout(() => {
        document.getElementById('renewalForm').reset();
        document.getElementById('renewalResults').style.display = 'none';
    }, 2000);
}

// Claims form handler
function initializeClaimsForm() {
    const form = document.getElementById('claimsForm');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="loading"></span> Submitting...';
        submitBtn.disabled = true;

        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 2000));

            const claimNumber = 'CLM' + Math.random().toString(36).substr(2, 9).toUpperCase();

            showNotification(
                `Claim submitted successfully! Your claim number is ${claimNumber}. An adjuster will contact you within 48 hours.`,
                'success'
            );

            form.reset();
        } catch (error) {
            showNotification('Failed to submit claim. Please try again.', 'error');
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });
}

// Show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
        ${message}
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// AI Vehicle Check Function (Customer Portal)
async function checkVehicleWithAI() {
    const btn = document.getElementById('aiCheckBtn');
    const resultDiv = document.getElementById('aiAssessmentResult');
    const assessmentText = document.getElementById('aiAssessmentText');
    const assessmentStatus = document.getElementById('aiAssessmentStatus');

    // Get vehicle info
    const make = document.getElementById('vehicleMake').value;
    const model = document.getElementById('vehicleModel').value;
    const year = document.getElementById('vehicleYear').value;
    const power = document.querySelector('input[name="power"]').value || 150;
    const fuelType = document.querySelector('select[name="type_fuel"]').value || 'P';

    // Validate inputs
    if (!make || !model || !year) {
        showNotification('Please enter vehicle make, model, and year', 'error');
        return;
    }

    // Show loading state
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Checking with AI...';
    resultDiv.style.display = 'none';

    try {
        // Call AI API
        const response = await fetch(`${API_V1}/llm/check-vehicle`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                make: make,
                model: model,
                year: parseInt(year),
                fuel_type: fuelType,
                power: parseInt(power),
                usage: 'personal',
                customer_age: 35  // Could get from step 1
            })
        });

        if (!response.ok) {
            throw new Error('AI service unavailable');
        }

        const data = await response.json();

        // Display result
        assessmentText.textContent = data.assessment;

        if (data.can_proceed_to_quote) {
            assessmentStatus.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <strong>Good News!</strong> ${data.message}
                </div>
            `;
        } else {
            assessmentStatus.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Notice:</strong> ${data.message}
                </div>
            `;
        }

        resultDiv.style.display = 'block';
        showNotification('AI assessment complete!', 'success');

    } catch (error) {
        console.error('AI check failed:', error);
        showNotification('AI service is temporarily unavailable. You can continue with the quote.', 'error');
    } finally {
        // Restore button
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-robot"></i> Check My Vehicle with AI';
    }
}
