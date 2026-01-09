<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Risk Assessment Tool</title>
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="assessment-card">
            <div class="card-header">
                <div class="logo">
                    <i class="fas fa-heartbeat"></i>
                </div>
                <h1>Diabetes Risk Assessment</h1>
                <p class="subtitle">Evaluate your diabetes risk based on multiple health indicators</p>
            </div>
            
            <div class="card-body">
                <form method="POST" action="" id="diabetesForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="age">
                                <i class="fas fa-user"></i> Age
                            </label>
                            <div class="input-with-icon">
                                <input type="number" id="age" name="age" 
                                       placeholder="Enter age" 
                                       min="1" max="120" step="1" required>
                                <span class="input-icon">years</span>
                            </div>
                            <div class="input-hint">Risk increases after age 45</div>
                        </div>
                        
                        <div class="form-group">
                            <label for="weight">
                                <i class="fas fa-weight"></i> Weight
                            </label>
                            <div class="input-with-icon">
                                <input type="number" id="weight" name="weight" 
                                       placeholder="Enter weight" 
                                       min="20" max="300" step="0.1" required>
                                <span class="input-icon">kg</span>
                            </div>
                            <div class="input-hint">Enter weight in kilograms</div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="bmi">
                            <i class="fas fa-weight-scale"></i> Body Mass Index (BMI)
                        </label>
                        <div class="input-with-icon">
                            <input type="number" id="bmi" name="bmi" 
                                   placeholder="Enter BMI" 
                                   min="15" max="60" step="0.1" required>
                            <span class="input-icon">kg/m²</span>
                        </div>
                        <div class="input-hint">Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: 30+</div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="systolic">
                                <i class="fas fa-heartbeat"></i> Systolic BP
                            </label>
                            <div class="input-with-icon">
                                <input type="number" id="systolic" name="systolic" 
                                       placeholder="Systolic" 
                                       min="70" max="250" step="1" required>
                                <span class="input-icon">mmHg</span>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="diastolic">
                                <i class="fas fa-heartbeat"></i> Diastolic BP
                            </label>
                            <div class="input-with-icon">
                                <input type="number" id="diastolic" name="diastolic" 
                                       placeholder="Diastolic" 
                                       min="40" max="150" step="1" required>
                                <span class="input-icon">mmHg</span>
                            </div>
                        </div>
                    </div>
                    <div class="input-hint" style="margin-top: -15px; margin-bottom: 25px;">Normal: <120/80 mmHg | Hypertension: ≥140/90 mmHg</div>
                    
                    <div class="form-group">
                        <label for="glucose">
                            <i class="fas fa-tint"></i> Fasting Glucose Level
                        </label>
                        <div class="input-with-icon">
                            <input type="number" id="glucose" name="glucose" 
                                   placeholder="Enter glucose level" 
                                   min="50" max="500" step="1" required>
                            <span class="input-icon">mg/dL</span>
                        </div>
                        <div class="input-hint">Normal: 70-100 mg/dL | Prediabetes: 100-125 mg/dL | Diabetes: ≥126 mg/dL</div>
                    </div>
                    
                    <button type="submit" class="btn-primary">
                        <i class="fas fa-stethoscope"></i> Assess Diabetes Risk
                    </button>
                    
                    <div class="form-footer">
                        <p><i class="fas fa-info-circle"></i> Enter your health information above to get your risk assessment</p>
                    </div>
                </form>
                
                <?php
                // PHP Processing
                if ($_SERVER['REQUEST_METHOD'] === 'POST') {
                    // Get form data
                    $age = isset($_POST['age']) ? intval($_POST['age']) : 0;
                    $weight = isset($_POST['weight']) ? floatval($_POST['weight']) : 0;
                    $bmi = isset($_POST['bmi']) ? floatval($_POST['bmi']) : 0;
                    $systolic = isset($_POST['systolic']) ? intval($_POST['systolic']) : 0;
                    $diastolic = isset($_POST['diastolic']) ? intval($_POST['diastolic']) : 0;
                    $glucose = isset($_POST['glucose']) ? floatval($_POST['glucose']) : 0;
                    
                    // Validate inputs
                    $errors = [];
                    if ($age <= 0 || $age > 120) $errors[] = "Age must be between 1-120 years";
                    if ($weight <= 0 || $weight > 300) $errors[] = "Weight must be between 1-300 kg";
                    if ($bmi <= 0 || $bmi > 60) $errors[] = "BMI must be between 1-60";
                    if ($systolic <= 0 || $systolic > 250) $errors[] = "Systolic BP must be between 70-250 mmHg";
                    if ($diastolic <= 0 || $diastolic > 150) $errors[] = "Diastolic BP must be between 40-150 mmHg";
                    if ($glucose <= 0 || $glucose > 500) $errors[] = "Glucose level must be between 50-500 mg/dL";
                    
                    if (empty($errors)) {
                        // Calculate risk score (enhanced algorithm)
                        $riskScore = 0;
                        $riskFactors = [];
                        
                        // Age risk factor
                        if ($age >= 45) { 
                            $riskScore += 1.5; 
                            $riskFactors[] = "Age 45 or above"; 
                        } elseif ($age >= 35) {
                            $riskScore += 0.5;
                        }
                        
                        // BMI risk factor
                        if ($bmi >= 30) { 
                            $riskScore += 2; 
                            $riskFactors[] = "High BMI (≥30)"; 
                        } elseif ($bmi >= 25) {
                            $riskScore += 1;
                            $riskFactors[] = "Overweight (BMI 25-29.9)";
                        }
                        
                        // Blood Pressure risk factor
                        $hasHypertension = false;
                        if ($systolic >= 140 || $diastolic >= 90) {
                            $riskScore += 1.5;
                            $riskFactors[] = "Hypertension (≥140/90 mmHg)";
                            $hasHypertension = true;
                        } elseif ($systolic >= 130 || $diastolic >= 85) {
                            $riskScore += 0.5;
                        }
                        
                        // Glucose risk factor
                        if ($glucose >= 126) { 
                            $riskScore += 2.5; 
                            $riskFactors[] = "Elevated glucose (≥126 mg/dL)"; 
                        } elseif ($glucose >= 100) {
                            $riskScore += 1;
                            $riskFactors[] = "Prediabetes range (100-125 mg/dL)";
                        }
                        
                        // Weight risk factor (additional)
                        if ($weight >= 90) {
                            $riskScore += 0.5;
                        }
                        
                        // Determine risk level based on total score
                        $totalRiskScore = $riskScore;
                        $riskLevel = '';
                        $title = '';
                        $icon = '';
                        $message = '';
                        $recommendations = [];
                        
                        if ($totalRiskScore >= 5) {
                            $riskLevel = "high";
                            $title = "High Risk Detected";
                            $icon = "fas fa-exclamation-triangle";
                            $message = "Based on your health indicators, you have a high risk for developing diabetes. Immediate consultation with a healthcare provider is recommended.";
                            $recommendations = [
                                "Schedule an appointment with your doctor immediately",
                                "Request comprehensive blood tests (HbA1c, lipid profile)",
                                "Adopt a strict diabetic-friendly diet",
                                "Begin regular exercise program (30 min/day, 5 days/week)",
                                "Monitor blood glucose and blood pressure daily",
                                "Consider weight management program",
                                "Avoid smoking and limit alcohol consumption"
                            ];
                        } elseif ($totalRiskScore >= 3) {
                            $riskLevel = "moderate-high";
                            $title = "Moderate to High Risk";
                            $icon = "fas fa-exclamation-circle";
                            $message = "You have multiple risk factors for diabetes. Preventive measures and lifestyle changes are strongly recommended.";
                            $recommendations = [
                                "Consult with a healthcare provider within the next month",
                                "Get fasting blood glucose and HbA1c tests",
                                "Start a balanced diet with controlled carbohydrates",
                                "Begin regular physical activity (150 min/week)",
                                "Monitor your weight and BMI monthly",
                                "Check blood pressure regularly",
                                "Reduce stress through relaxation techniques"
                            ];
                        } elseif ($totalRiskScore >= 1.5) {
                            $riskLevel = "moderate";
                            $title = "Moderate Risk";
                            $icon = "fas fa-info-circle";
                            $message = "You have some risk factors for diabetes. Maintaining a healthy lifestyle is important for prevention.";
                            $recommendations = [
                                "Schedule a routine health check-up",
                                "Maintain healthy weight through balanced diet",
                                "Exercise regularly (at least 30 min most days)",
                                "Monitor blood glucose annually",
                                "Eat more fruits, vegetables, and whole grains",
                                "Limit processed foods and sugary drinks",
                                "Get adequate sleep (7-8 hours nightly)"
                            ];
                        } else {
                            $riskLevel = "low";
                            $title = "Low Risk";
                            $icon = "fas fa-check-circle";
                            $message = "Based on your current health indicators, you have a low risk for diabetes. Continue maintaining healthy habits.";
                            $recommendations = [
                                "Continue with your healthy lifestyle",
                                "Maintain regular physical activity",
                                "Eat a balanced, nutritious diet",
                                "Get annual health screenings",
                                "Stay hydrated with water",
                                "Manage stress effectively",
                                "Maintain healthy weight range"
                            ];
                        }
                        
                        $risk_factors_text = implode(", ", $riskFactors);
                        if (empty($risk_factors_text)) {
                            $risk_factors_text = "No significant risk factors identified";
                        }
                        
                        // Display result
                        echo '
                        <div class="result-container ' . $riskLevel . '-risk">
                            <div class="result-header">
                                <div class="result-icon">
                                    <i class="' . $icon . '"></i>
                                </div>
                                <h2>' . $title . '</h2>
                                <p class="risk-score">Risk Score: ' . number_format($totalRiskScore, 1) . '/8</p>
                            </div>
                            
                            <div class="result-body">
                                <p class="result-message">' . $message . '</p>
                                
                                <div class="health-summary">
                                    <h3><i class="fas fa-clipboard-list"></i> Your Health Summary</h3>
                                    <div class="summary-grid">
                                        <div class="summary-item">
                                            <span class="summary-label">Age:</span>
                                            <span class="summary-value">' . $age . ' years</span>
                                        </div>
                                        <div class="summary-item">
                                            <span class="summary-label">Weight:</span>
                                            <span class="summary-value">' . $weight . ' kg</span>
                                        </div>
                                        <div class="summary-item">
                                            <span class="summary-label">BMI:</span>
                                            <span class="summary-value">' . $bmi . ' kg/m²</span>
                                        </div>
                                        <div class="summary-item">
                                            <span class="summary-label">Blood Pressure:</span>
                                            <span class="summary-value">' . $systolic . '/' . $diastolic . ' mmHg</span>
                                        </div>
                                        <div class="summary-item">
                                            <span class="summary-label">Glucose:</span>
                                            <span class="summary-value">' . $glucose . ' mg/dL</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="risk-details">
                                    <h3><i class="fas fa-chart-line"></i> Risk Assessment Details</h3>
                                    <div class="details-grid">
                                        <div class="detail-item">
                                            <span class="detail-label">Risk Level:</span>
                                            <span class="detail-value ' . $riskLevel . '">' . ucwords(str_replace('-', ' ', $riskLevel)) . ' Risk</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-label">Identified Factors:</span>
                                            <span class="detail-value">' . htmlspecialchars($risk_factors_text) . '</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="recommendations">
                                    <h3><i class="fas fa-lightbulb"></i> Personalized Recommendations</h3>
                                    <ul>';
                                    
                        foreach ($recommendations as $rec) {
                            echo '<li><i class="fas fa-check"></i> ' . htmlspecialchars($rec) . '</li>';
                        }
                        
                        echo '
                                    </ul>
                                </div>
                                
                                <div class="result-actions">
                                    <button type="button" class="new-assessment-btn" onclick="clearForm()">
                                        <i class="fas fa-redo"></i> New Assessment
                                    </button>
                                    <button class="btn-secondary" onclick="window.print()">
                                        <i class="fas fa-print"></i> Print Results
                                    </button>
                                </div>
                                
                                <div class="disclaimer-note">
                                    <p><i class="fas fa-exclamation-circle"></i> <strong>Important:</strong> This assessment is for informational purposes only. Always consult with a healthcare professional for accurate diagnosis and treatment.</p>
                                </div>
                            </div>
                        </div>';
                    } else {
                        // Display errors
                        echo '<div class="error-message">
                                <h3><i class="fas fa-exclamation-triangle"></i> Please correct the following:</h3>
                                <ul>';
                        foreach ($errors as $error) {
                            echo '<li>' . htmlspecialchars($error) . '</li>';
                        }
                        echo '</ul></div>';
                    }
                }
                ?>
                
                <div class="info-section">
                    <div class="info-card">
                        <h3><i class="fas fa-clipboard-check"></i> About This Assessment</h3>
                        <p>This comprehensive tool evaluates diabetes risk based on multiple clinical indicators including age, weight, BMI, blood pressure, and fasting glucose levels.</p>
                    </div>
                    
                    <div class="info-card">
                        <h3><i class="fas fa-shield-alt"></i> Privacy & Security</h3>
                        <p>Your health data is processed securely and not stored on our servers. All calculations are done in real-time and results disappear when you refresh the page.</p>
                    </div>
                </div>
            </div>
            
            <div class="card-footer">
                <p><i class="fas fa-exclamation-triangle"></i> <strong>Disclaimer:</strong> This tool is for educational purposes only and not a substitute for professional medical advice, diagnosis, or treatment.</p>
                <p class="copyright">© <?php echo date('Y'); ?> Diabetes Risk Assessment Tool</p>
            </div>
        </div>
    </div>
    
    <script src="script.js"></script>
    <script>
        // Clear form function for New Assessment button
        function clearForm() {
            document.getElementById('diabetesForm').reset();
            
            // Remove result container
            const resultContainer = document.querySelector('.result-container');
            if (resultContainer) {
                resultContainer.remove();
            }
            
            // Remove error messages
            const errorMessages = document.querySelectorAll('.error-message');
            errorMessages.forEach(error => error.remove());
            
            // Remove input styling
            const inputs = document.querySelectorAll('input');
            inputs.forEach(input => {
                input.classList.remove('input-low', 'input-normal', 'input-high', 'input-error');
            });
            
            // Scroll to form
            document.querySelector('form').scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Focus on first input
            setTimeout(() => {
                document.getElementById('age').focus();
            }, 500);
        }
        
        // Real-time input validation and styling
        document.addEventListener('DOMContentLoaded', function() {
            const inputs = document.querySelectorAll('input[type="number"]');
            
            inputs.forEach(input => {
                input.addEventListener('input', function() {
                    validateAndStyleInput(this);
                });
                
                input.addEventListener('focus', function() {
                    this.select();
                });
            });
            
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
            
            // Focus on first input when page loads
            const firstInput = document.getElementById('age');
            if (firstInput) {
                setTimeout(() => {
                    firstInput.focus();
                }, 500);
            }
        });
    </script>
</body>
</html>