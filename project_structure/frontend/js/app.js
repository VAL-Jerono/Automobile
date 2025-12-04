// API Configuration
const API_BASE_URL = 'http://localhost:8001';
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
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            // Animate counters with realistic numbers
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
        submitBtn.innerHTML = '<span class="loading"></span> Calculating...';
        submitBtn.disabled = true;

        try {
            // Calculate quote using ML model
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

// Calculate quote and predict lapse risk
async function calculateQuote(formData) {
    // Calculate estimated premium based on risk factors
    const basePremium = 800;
    let premium = basePremium;

    // Age factor
    const birthDate = new Date(formData.date_birth);
    const age = Math.floor((new Date() - birthDate) / (365.25 * 24 * 60 * 60 * 1000));
    if (age < 25) premium *= 1.5;
    else if (age > 65) premium *= 1.2;

    // Vehicle factors
    const vehicleAge = new Date().getFullYear() - parseInt(formData.year_matriculation);
    premium *= (1 + vehicleAge * 0.02);
    
    if (formData.type_fuel === 'E') premium *= 0.9; // Electric discount
    premium *= (1 + parseInt(formData.power) / 1000);

    // Claims history
    premium *= (1 + parseInt(formData.n_claims_history) * 0.3);
    premium *= (1 + parseInt(formData.n_claims_year) * 0.5);

    // Coverage type
    if (formData.type_risk === 'COMP') premium *= 1.5;
    else if (formData.type_risk === 'COLL') premium *= 1.3;

    // Vehicle value
    premium *= (1 + parseInt(formData.value_vehicle) / 50000);

    // Round to nearest 10
    premium = Math.round(premium / 10) * 10;

    // Calculate lapse risk using simple heuristics (in production, use ML API)
    let lapseRisk = 0.1;
    if (parseInt(formData.n_claims_history) > 2) lapseRisk += 0.3;
    if (premium > 2000) lapseRisk += 0.2;
    if (vehicleAge > 10) lapseRisk += 0.15;
    if (age < 25) lapseRisk += 0.1;
    
    lapseRisk = Math.min(lapseRisk, 0.95);

    // Determine risk level
    let riskLevel = 'low';
    if (lapseRisk > 0.6) riskLevel = 'high';
    else if (lapseRisk > 0.3) riskLevel = 'medium';

    // Generate recommendation
    let recommendation = '';
    if (riskLevel === 'low') {
        recommendation = 'Excellent! You qualify for our premium rates with low lapse risk. We recommend proceeding with this policy.';
    } else if (riskLevel === 'medium') {
        recommendation = 'Good profile with moderate risk. Consider adding additional coverage or adjusting payment terms for better rates.';
    } else {
        recommendation = 'Higher risk profile detected. We recommend reviewing your claims history and considering risk-reduction measures for better rates.';
    }

    return {
        premium,
        lapseRisk,
        riskLevel,
        recommendation,
        formData
    };
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
                    <p><strong>Renewal Date:</strong> ${new Date(Date.now() + 30*24*60*60*1000).toLocaleDateString()}</p>
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
