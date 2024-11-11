window.addEventListener('load', () => {
    // Intro text for typing animation
    const introText = "Welcome to DataNow! \n Your Data ChatBot.";
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

$(document).ready(function() {
    const sessionId = "{{ session.sid }}";

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingIndicator = `<div id="typing-indicator" class="system">
                                    Typing<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
                                 </div>`;
        $('#chat-history').append(typingIndicator);
        $('#chat-history').scrollTop($('#chat-history')[0].scrollHeight);

        // Animate the dots for a typing effect
        const dots = document.querySelectorAll("#typing-indicator .dot");
        let dotIndex = 0;
        setInterval(() => {
            dots.forEach(dot => dot.style.opacity = "0.2");
            dots[dotIndex].style.opacity = "1";
            dotIndex = (dotIndex + 1) % dots.length;
        }, 500); // Adjust the speed for dot animation
    }

    // Function to hide typing indicator
    function hideTypingIndicator() {
        $('#typing-indicator').remove();
    }

    // Load chat history
    $.get(`/get_chat_history?session_id=${sessionId}`, function(data) {
        for (const msg of data) {
            $('#chat-history').append(`<div class="${msg.role}">${msg.message}</div>`);
        }
    });

    // File upload handler
    $('#file-upload').on('change', function() {
        const file = this.files[0];
        if (file) {
            let formData = new FormData();
            formData.append("file", file);

            $.ajax({
                url: "/upload",
                type: "POST",
                data: formData,
                processData: false,
                contentType: false,
                success: function(response) {
                    $('#chat-history').append(`<div class="system">${response.message}</div>`);
                },
                error: function(jqXHR) {
                    $('#chat-history').append(`<div class="system">Error: ${jqXHR.responseJSON.error}</div>`);
                }
            });
        } else {
            $('#chat-history').append(`<div class="system">Please select a file first.</div>`);
        }
    });

    // Submit prompt handler
    $('#submit-prompt').click(function() {
        const prompt = $('#user_prompt').val();
        if (prompt) {
            $('#chat-history').append(`<div class="user">${prompt}</div>`);
            $('#user_prompt').val('');

            // Show typing indicator
            showTypingIndicator();

            // Send the prompt to OpenAI
            $.post("/ask_openai", { user_input: prompt, session_id: sessionId }, function(response) {
                hideTypingIndicator();  // Hide typing indicator once the response is received
                if (response.html) {
                    $('#chat-history').append(`<div class="system">${response.html}</div>`);
                } else if (response.image) {
                    $('#chat-history').append(`<div class="system"><img src="${response.image}" alt="Generated Graph" class="generated-graph"></div>`);
                } else {
                    $('#chat-history').append(`<div class="system">${response.message}</div>`);
                }
                $('#chat-history').scrollTop($('#chat-history')[0].scrollHeight);
            }).fail(function(jqXHR) {
                hideTypingIndicator();  // Hide typing indicator in case of an error
                $('#chat-history').append(`<div class="system">Error: ${jqXHR.responseJSON.error}</div>`);
            });
        } else {
            $('#chat-history').append(`<div class="system">Please enter a message.</div>`);
        }
    });

    // Download PDF handler
    $('#download-pdf').click(function() {
        window.location.href = "/download_pdf?session_id=" + sessionId;
    });
});
