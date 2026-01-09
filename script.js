document.addEventListener('DOMContentLoaded', function() {
    // Real-time input validation and styling
    initializeInputValidation();
    
    // Add keyboard shortcut for new assessment (Ctrl + N)
    document.addEventListener('keydown', function(event) {
        if ((event.ctrlKey || event.metaKey) && event.key === 'n') {
            event.preventDefault();
            const newAssessmentBtn = document.querySelector('.new-assessment-btn');
            if (newAssessmentBtn) {
                newAssessmentBtn.click();
            }
        }
    });
    
    // BMI Calculator (optional - could calculate from height/weight)
    setupBMICalculator();
});

function initializeInputValidation() {
    const inputs = document.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateAndStyleInput(this);
            showInputHint(this);
        });
        
        input.addEventListener('blur', function() {
            showInputHint(this);
        });
        
        input.addEventListener('focus', function() {
            this.select();
            hideInputHint(this);
        });
    });
}

function validateAndStyleInput(input) {
    const value = parseFloat(input.value);
    const id = input.id;
    
    // Clear previous styles
    input.classList.remove('input-low', 'input-normal', 'input-high', 'input-error');
    
    if (isNaN(value)) return;
    
    // Apply color coding based on value ranges
    if (id === 'age') {
        if (value >= 45) {
            input.classList.add('input-high');
        } else if (value >= 35) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    } else if (id === 'weight') {
        if (value >= 90) {
            input.classList.add('input-high');
        } else if (value >= 70) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    } else if (id === 'bmi') {
        if (value >= 30) {
            input.classList.add('input-high');
        } else if (value >= 25) {
            input.classList.add('input-high');
        } else if (value >= 18.5) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    } else if (id === 'systolic') {
        if (value >= 140) {
            input.classList.add('input-high');
        } else if (value >= 130) {
            input.classList.add('input-normal');
        } else if (value >= 120) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    } else if (id === 'diastolic') {
        if (value >= 90) {
            input.classList.add('input-high');
        } else if (value >= 85) {
            input.classList.add('input-normal');
        } else if (value >= 80) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    } else if (id === 'glucose') {
        if (value >= 126) {
            input.classList.add('input-high');
        } else if (value >= 100) {
            input.classList.add('input-high');
        } else if (value >= 70) {
            input.classList.add('input-normal');
        } else {
            input.classList.add('input-low');
        }
    }
}

function showInputHint(input) {
    const value = parseFloat(input.value);
    const id = input.id;
    
    if (isNaN(value)) {
        hideInputHint(input);
        return;
    }
    
    let hintText = '';
    
    switch(id) {
        case 'age':
            if (value >= 45) hintText = 'Higher risk age group';
            else if (value >= 35) hintText = 'Moderate risk age group';
            else hintText = 'Lower risk age group';
            break;
            
        case 'weight':
            if (value >= 90) hintText = 'High weight - increased risk';
            else if (value >= 70) hintText = 'Normal weight range';
            else hintText = 'Low weight';
            break;
            
        case 'bmi':
            if (value >= 30) hintText = 'Obese - high risk';
            else if (value >= 25) hintText = 'Overweight - increased risk';
            else if (value >= 18.5) hintText = 'Normal weight';
            else hintText = 'Underweight';
            break;
            
        case 'systolic':
            if (value >= 140) hintText = 'Stage 2 Hypertension';
            else if (value >= 130) hintText = 'Stage 1 Hypertension';
            else if (value >= 120) hintText = 'Elevated';
            else hintText = 'Normal';
            break;
            
        case 'diastolic':
            if (value >= 90) hintText = 'Stage 2 Hypertension';
            else if (value >= 85) hintText = 'Stage 1 Hypertension';
            else if (value >= 80) hintText = 'Elevated';
            else hintText = 'Normal';
            break;
            
        case 'glucose':
            if (value >= 126) hintText = 'Diabetes range';
            else if (value >= 100) hintText = 'Prediabetes range';
            else if (value >= 70) hintText = 'Normal glucose level';
            else hintText = 'Low glucose level';
            break;
    }
    
    // Create or update hint element
    let hintElement = input.parentNode.querySelector('.real-time-hint');
    if (!hintElement) {
        hintElement = document.createElement('div');
        hintElement.className = 'real-time-hint';
        input.parentNode.appendChild(hintElement);
    }
    
    hintElement.textContent = hintText;
    hintElement.style.display = 'block';
}

function hideInputHint(input) {
    const hintElement = input.parentNode.querySelector('.real-time-hint');
    if (hintElement) {
        hintElement.style.display = 'none';
    }
}

function setupBMICalculator() {
    // Optional: Add BMI calculator if needed
    // This could calculate BMI from height and weight if we add height input
}

// Function to clear the form (called from New Assessment button)
function clearForm() {
    const form = document.getElementById('diabetesForm');
    if (form) {
        form.reset();
        
        // Remove result container
        const resultContainer = document.querySelector('.result-container');
        if (resultContainer) {
            resultContainer.remove();
        }
        
        // Remove error messages
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(error => error.remove());
        
        // Remove input styling and hints
        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
            input.classList.remove('input-low', 'input-normal', 'input-high', 'input-error');
            hideInputHint(input);
        });
        
        // Scroll to form
        form.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Focus on first input
        setTimeout(() => {
            document.getElementById('age').focus();
        }, 500);
    }
}

// Add CSS for real-time hints
const style = document.createElement('style');
style.textContent = `
    .real-time-hint {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
        padding-left: 5px;
        display: none;
        animation: fadeIn 0.3s;
        font-style: italic;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
`;
document.head.appendChild(style);