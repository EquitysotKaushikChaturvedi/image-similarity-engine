const searchBtn = document.getElementById('searchBtn');
const imageInput = document.getElementById('imageInput');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');

searchBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];

    if (!file) {
        alert("PLEASE SELECT AN IMAGE FIRST.");
        return;
    }

    // Show loading state
    searchBtn.textContent = "SEARCHING...";
    searchBtn.disabled = true;
    resultsGrid.innerHTML = '';
    resultsSection.classList.remove('hidden');

    const limit = document.getElementById('limitSelect').value;

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Pass topk parameter to backend
        const response = await fetch(`/search?topk=${limit}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Render results
        let visibleCount = 0;

        data.matches.forEach(match => {
            // User requested "Accurate" results.
            // If the score is too low (e.g., below 80%), it's likely just noise (random cats/dogs).
            // We filter those out to keep the quality high.
            if (match.score < 0.80) {
                return;
            }

            visibleCount++;
            const card = document.createElement('div');
            card.className = 'result-card';

            // Note: We need the backend to serve images too for this to work perfectly!
            // For now we'll assume the filenames are relative to the 'data' folder served at /images

            // Constructing string for score
            const percent = (match.score * 100).toFixed(1) + '%';

            card.innerHTML = `
                <img src="/images/${match.filename}" alt="${match.filename}">
                <span>REF: ${match.filename}<br>SCORE: ${percent}</span>
            `;

            resultsGrid.appendChild(card);
        });

        if (visibleCount === 0) {
            resultsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center;">NO ACCURATE MATCHES FOUND (>80%).</div>';
        }

    } catch (error) {
        console.error('Error:', error);
        resultsGrid.innerHTML = `<p style="color:red; grid-column: 1/-1;">ERROR: ${error.message}</p>`;
    } finally {
        searchBtn.textContent = "SEARCH_DATABASE [ENTER]";
        searchBtn.disabled = false;
    }
});
