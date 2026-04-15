document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // UI Elements
    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('btnLoader');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const cardsContainer = document.getElementById('cardsContainer');
    
    // Gather values
    const payload = {
        n: document.getElementById('n').value,
        p: document.getElementById('p').value,
        k: document.getElementById('k').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        rainfall: document.getElementById('rainfall').value
    };

    // Show loading state
    btnText.textContent = "Analyzing...";
    loader.classList.remove('hidden');
    submitBtn.style.opacity = '0.8';
    submitBtn.disabled = true;
    
    // Hide previous results
    resultsSection.classList.add('hidden');
    cardsContainer.innerHTML = '';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Generate Cards
        data.forEach((item, index) => {
            let statusClass = '';
            let tagClass = '';
            
            if (item.risk === 'low') {
                statusClass = 'status-best';
                tagClass = 'tag-best';
            } else if (item.risk === 'medium') {
                statusClass = 'status-mod';
                tagClass = 'tag-mod';
            } else {
                statusClass = 'status-risk';
                tagClass = 'tag-risk';
            }

            // Formatting currency
            const formatter = new Intl.NumberFormat('en-IN', {
                style: 'currency',
                currency: 'INR',
                maximumFractionDigits: 0
            });

            const card = document.createElement('div');
            card.className = `crop-card ${statusClass}`;
            card.style.animationDelay = `${index * 0.15}s`;
            card.style.animation = `fadeInUp 0.6s ease forwards`;
            card.style.opacity = '0';
            
            card.innerHTML = `
                <div class="card-header">
                    <span class="crop-name">${item.crop}</span>
                    <span class="rank-badge">#${index + 1}</span>
                </div>
                
                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Est. Profit</span>
                        <span class="stat-value profit-value">${formatter.format(item.profit)}/ha</span>
                    </div>
                    
                    <div class="stat-row">
                        <span class="stat-label">Risk Level</span>
                        <span class="stat-value" style="text-transform: capitalize;">${item.risk}</span>
                    </div>
                    
                    <div class="tag ${tagClass}">
                        ${item.recommendation}
                    </div>
                </div>
            `;
            
            cardsContainer.appendChild(card);
        });

        // Show results
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error('Error fetching prediction:', error);
        alert("Failed to connect to the backend API. Ensure the Flask server is running on http://127.0.0.1:5000");
    } finally {
        // Reset loading state
        btnText.textContent = "Get Recommendation";
        loader.classList.add('hidden');
        submitBtn.style.opacity = '1';
        submitBtn.disabled = false;
    }
});

// Auto-fill Weather Logic
document.getElementById('autoFillBtn').addEventListener('click', async function() {
    const btn = this;
    const btnText = document.getElementById('autoFillText');
    const originalText = btnText.innerHTML;
    
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser.");
        return;
    }
    
    btnText.textContent = "Locating... ⏳";
    btn.disabled = true;
    
    navigator.geolocation.getCurrentPosition(async (position) => {
        try {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            btnText.textContent = "Fetching weather... ⛅";
            
            const response = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`);
            if (!response.ok) throw new Error("Weather API failed");
            
            const data = await response.json();
            
            document.getElementById('temperature').value = data.current.temperature_2m;
            document.getElementById('humidity').value = data.current.relative_humidity_2m;
            
            btnText.textContent = "Weather Updated! ✔️";
            setTimeout(() => {
                btnText.innerHTML = originalText;
                btn.disabled = false;
            }, 3000);
            
        } catch (error) {
            console.error(error);
            alert("Failed to fetch weather data.");
            btnText.innerHTML = originalText;
            btn.disabled = false;
        }
    }, (error) => {
        alert("Unable to retrieve your location. " + error.message);
        btnText.innerHTML = originalText;
        btn.disabled = false;
    });
});

