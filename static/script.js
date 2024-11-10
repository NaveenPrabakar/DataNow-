window.addEventListener('load', () => {
    // Intro text for typing animation
    const introText = "Welcome to DataNow! Your Data ChatBot.";
    const introElement = document.getElementById("intro-text");

    // Type out the text one character at a time
    let index = 0;
    function typeEffect() {
        if (index < introText.length) {
            introElement.innerHTML += introText.charAt(index);
            index++;
            setTimeout(typeEffect, 100); // Adjust speed if needed
        } else {
            setTimeout(() => {
                // After typing animation, hide animation container and show chat
                document.getElementById("animation-container").style.display = "none";
                document.getElementById("chat-container").style.display = "flex";
            }, 1000); // Pause briefly before revealing chat
        }
    }

    typeEffect(); // Start typing effect
});
