document.addEventListener('DOMContentLoaded', () => {
    const translateForm = document.getElementById('translateForm');
    const translationResult = document.getElementById('translationResult');
    const translatedText = document.getElementById('translatedText');
    const errorMessage = document.getElementById('errorMessage');

    translateForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const urlInput = document.getElementById('url').value;
        const languageInput = document.getElementById('language').value;

        translationResult.style.display = 'none';
        errorMessage.style.display = 'none';

        try {
            const response = await fetch(`/translate?url=${encodeURIComponent(urlInput)}&lang=${encodeURIComponent(languageInput)}`);
            const data = await response.json();

            if (response.ok) {
                translatedText.innerText = data.translation;
                translationResult.style.display = 'block';
            } else {
                errorMessage.innerText = data.error || 'An error occurred during translation.';
                errorMessage.style.display = 'block';
            }
        } catch (error) {
            errorMessage.innerText = 'Failed to fetch the translation.';
            errorMessage.style.display = 'block';
        }
    });
});
