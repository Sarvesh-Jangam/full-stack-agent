import re
from typing import Dict, Any
from google.adk.agents import Agent

def generate_html_component(description: str, requirements: str = "") -> Dict[str, Any]:
    """
    Generates a complete HTML file with Tailwind CSS and JavaScript based on user description.

    Args:
        description (str): Description of the web component/application to create
        requirements (str): Additional specific requirements or constraints

    Returns:
        Dict[str, Any]: Contains status, generated HTML code, and explanation
    """
    try:
        # This is a placeholder function that would normally generate HTML
        # In a real implementation, this would use the LLM to generate code

        if not description.strip():
            return {
                "status": "error",
                "error_message": "Please provide a description of what you want to create."
            }

        # Simulate HTML generation based on common patterns
        if "todo" in description.lower() or "task" in description.lower():
            html_code = generate_todo_app()
        elif "landing" in description.lower() or "homepage" in description.lower():
            html_code = generate_landing_page()
        elif "form" in description.lower():
            html_code = generate_contact_form()
        else:
            html_code = generate_basic_template(description)

        return {
            "status": "success",
            "html_code": html_code,
            "explanation": f"Generated a responsive HTML application for: {description}",
            "features": [
                "Semantic HTML5 structure",
                "Tailwind CSS for styling",
                "Responsive design",
                "Modern JavaScript (ES6+)",
                "Clean and maintainable code"
            ]
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error generating HTML: {str(e)}"
        }

def validate_and_optimize_code(html_code: str) -> Dict[str, Any]:
    """
    Validates and suggests optimizations for HTML/CSS/JS code.

    Args:
        html_code (str): HTML code to validate

    Returns:
        Dict[str, Any]: Validation results and optimization suggestions
    """
    try:
        suggestions = []

        # Basic HTML validation
        if not html_code.strip():
            return {
                "status": "error",
                "error_message": "No HTML code provided for validation."
            }

        # Check for basic HTML structure
        if "<!DOCTYPE html>" not in html_code:
            suggestions.append("Add DOCTYPE declaration for HTML5")

        if "lang=" not in html_code:
            suggestions.append("Add language attribute to html tag")

        if "viewport" not in html_code:
            suggestions.append("Add viewport meta tag for responsive design")

        # Check for Tailwind CSS
        if "tailwindcss.com" not in html_code:
            suggestions.append("Consider including Tailwind CSS CDN for styling")

        # Check for accessibility
        if "alt=" not in html_code and "<img" in html_code:
            suggestions.append("Add alt attributes to images for accessibility")

        return {
            "status": "success",
            "is_valid": len(suggestions) == 0,
            "suggestions": suggestions,
            "code_quality": "Good" if len(suggestions) <= 2 else "Needs improvement"
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error validating code: {str(e)}"
        }

def generate_todo_app():
    """Generate a complete todo application"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Todo List App</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen py-8">
    <div class="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 class="text-2xl font-bold text-gray-800 mb-6 text-center">My Todo List</h1>

        <!-- Add todo form -->
        <div class="mb-6">
            <div class="flex gap-2">
                <input 
                    type="text" 
                    id="todoInput" 
                    placeholder="Add a new task..."
                    class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                <button 
                    onclick="addTodo()"
                    class="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    Add
                </button>
            </div>
        </div>

        <!-- Todo list -->
        <ul id="todoList" class="space-y-2">
            <!-- Todos will be added here dynamically -->
        </ul>

        <!-- Stats -->
        <div class="mt-6 text-sm text-gray-600 text-center">
            <span id="todoCount">0</span> tasks remaining
        </div>
    </div>

    <script>
        let todos = [];
        let todoIdCounter = 1;

        function addTodo() {
            const input = document.getElementById('todoInput');
            const text = input.value.trim();

            if (text === '') return;

            const todo = {
                id: todoIdCounter++,
                text: text,
                completed: false
            };

            todos.push(todo);
            input.value = '';
            renderTodos();
        }

        function toggleTodo(id) {
            const todo = todos.find(t => t.id === id);
            if (todo) {
                todo.completed = !todo.completed;
                renderTodos();
            }
        }

        function deleteTodo(id) {
            todos = todos.filter(t => t.id !== id);
            renderTodos();
        }

        function renderTodos() {
            const todoList = document.getElementById('todoList');
            const todoCount = document.getElementById('todoCount');

            todoList.innerHTML = '';

            todos.forEach(todo => {
                const li = document.createElement('li');
                li.className = 'flex items-center gap-2 p-2 border border-gray-200 rounded-md';

                li.innerHTML = `
                    <input 
                        type="checkbox" 
                        ${todo.completed ? 'checked' : ''}
                        onchange="toggleTodo(${todo.id})"
                        class="w-4 h-4 text-blue-600"
                    >
                    <span class="flex-1 ${todo.completed ? 'line-through text-gray-500' : 'text-gray-800'}">${todo.text}</span>
                    <button 
                        onclick="deleteTodo(${todo.id})"
                        class="px-2 py-1 text-red-600 hover:bg-red-50 rounded"
                    >
                        Delete
                    </button>
                `;

                todoList.appendChild(li);
            });

            const remaining = todos.filter(t => !t.completed).length;
            todoCount.textContent = remaining;
        }

        // Allow adding todos with Enter key
        document.getElementById('todoInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                addTodo();
            }
        });
    </script>
</body>
</html>"""

def generate_landing_page():
    """Generate a basic landing page"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Our App</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <h1 class="text-xl font-bold text-gray-900">YourApp</h1>
                </div>
                <div class="flex items-center space-x-4">
                    <a href="#features" class="text-gray-600 hover:text-gray-900">Features</a>
                    <a href="#contact" class="text-gray-600 hover:text-gray-900">Contact</a>
                    <button class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                        Get Started
                    </button>
                </div>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="py-20">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 class="text-5xl font-bold text-gray-900 mb-6">
                Welcome to the Future
            </h2>
            <p class="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Discover amazing features that will transform the way you work and live. 
                Join thousands of users who have already made the switch.
            </p>
            <div class="space-x-4">
                <button class="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg hover:bg-blue-700 transition duration-200">
                    Start Free Trial
                </button>
                <button class="border border-blue-600 text-blue-600 px-8 py-3 rounded-lg text-lg hover:bg-blue-50 transition duration-200">
                    Learn More
                </button>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section id="features" class="py-20 bg-white">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center mb-16">
                <h3 class="text-3xl font-bold text-gray-900 mb-4">Amazing Features</h3>
                <p class="text-lg text-gray-600">Everything you need to succeed</p>
            </div>

            <div class="grid md:grid-cols-3 gap-8">
                <div class="text-center">
                    <div class="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                        </svg>
                    </div>
                    <h4 class="text-xl font-semibold text-gray-900 mb-2">Lightning Fast</h4>
                    <p class="text-gray-600">Experience blazing fast performance with our optimized platform.</p>
                </div>

                <div class="text-center">
                    <div class="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <h4 class="text-xl font-semibold text-gray-900 mb-2">Reliable</h4>
                    <p class="text-gray-600">99.9% uptime guaranteed with enterprise-grade infrastructure.</p>
                </div>

                <div class="text-center">
                    <div class="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg class="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                        </svg>
                    </div>
                    <h4 class="text-xl font-semibold text-gray-900 mb-2">Secure</h4>
                    <p class="text-gray-600">Bank-level security to keep your data safe and protected.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- CTA Section -->
    <section class="py-20 bg-blue-600">
        <div class="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
            <h3 class="text-3xl font-bold text-white mb-4">Ready to Get Started?</h3>
            <p class="text-xl text-blue-100 mb-8">Join thousands of satisfied users today</p>
            <button class="bg-white text-blue-600 px-8 py-3 rounded-lg text-lg font-semibold hover:bg-gray-50 transition duration-200">
                Start Your Free Trial
            </button>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center">
                <h4 class="text-lg font-semibold mb-4">YourApp</h4>
                <p class="text-gray-400">Making the world a better place through technology.</p>
            </div>
        </div>
    </footer>
</body>
</html>"""

def generate_contact_form():
    """Generate a contact form"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Us</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen py-12">
    <div class="max-w-md mx-auto bg-white rounded-lg shadow-md p-8">
        <h1 class="text-2xl font-bold text-gray-900 mb-6 text-center">Contact Us</h1>

        <form id="contactForm" class="space-y-4">
            <div>
                <label for="name" class="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input 
                    type="text" 
                    id="name" 
                    name="name" 
                    required
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
            </div>

            <div>
                <label for="email" class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input 
                    type="email" 
                    id="email" 
                    name="email" 
                    required
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
            </div>

            <div>
                <label for="message" class="block text-sm font-medium text-gray-700 mb-1">Message</label>
                <textarea 
                    id="message" 
                    name="message" 
                    rows="4" 
                    required
                    class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                ></textarea>
            </div>

            <button 
                type="submit"
                class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200"
            >
                Send Message
            </button>
        </form>

        <div id="successMessage" class="hidden mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-md">
            Thank you! Your message has been sent successfully.
        </div>
    </div>

    <script>
        document.getElementById('contactForm').addEventListener('submit', function(e) {
            e.preventDefault();

            // Simulate form submission
            const form = this;
            const submitButton = form.querySelector('button[type="submit"]');
            const successMessage = document.getElementById('successMessage');

            // Disable submit button and show loading state
            submitButton.disabled = true;
            submitButton.textContent = 'Sending...';

            // Simulate API call delay
            setTimeout(() => {
                // Show success message
                successMessage.classList.remove('hidden');

                // Reset form
                form.reset();

                // Reset button
                submitButton.disabled = false;
                submitButton.textContent = 'Send Message';

                // Hide success message after 5 seconds
                setTimeout(() => {
                    successMessage.classList.add('hidden');
                }, 5000);
            }, 1000);
        });
    </script>
</body>
</html>"""

def generate_basic_template(description):
    """Generate a basic HTML template"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Custom Application</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-6">Custom Application</h1>
        <div class="bg-white rounded-lg shadow-md p-6">
            <p class="text-gray-600 mb-4">
                This is a custom application generated based on your request: "{description}"
            </p>
            <p class="text-sm text-gray-500">
                This is a basic template. For more specific functionality, please provide more detailed requirements.
            </p>
        </div>
    </div>

    <script>
        // Custom JavaScript functionality would go here
        console.log('Application initialized');
    </script>
</body>
</html>"""

# Create the main ADK agent
root_agent = Agent(
    name="frontend_agent",
    model="gemini-1.5-flash-latest",
    description=(
        "Frontend Agent specialized in modern web development. Expert in generating clean, "
        "responsive HTML, CSS (Tailwind), and JavaScript code based on user requests."
    ),
    instruction=(
        "You are the Frontend Agent, a specialized AI assistant expert in modern frontend web development. "
        "Your primary goal is to generate clean, responsive, and aesthetically pleasing HTML, CSS using "
        "Tailwind CSS, and JavaScript code based on user requests. You are a sub-agent in a multi-agent "
        "system, and you will receive tasks from a Coordinator Agent. "
        ""
        "Core Capabilities: "
        "- HTML Structure: Generate semantic and well-structured HTML5 "
        "- Styling with Tailwind CSS: Use Tailwind CSS for all styling, create modern responsive layouts "
        "- JavaScript for Interactivity: Write clean, modern JavaScript (ES6+) for user interactions "
        "- Component-Based Design: Think in terms of reusable components "
        "- Code Quality: Generate well-commented, easy to understand code following best practices "
        ""
        "Constraints: "
        "- Technology Stack: Use HTML, Tailwind CSS, and vanilla JavaScript only "
        "- File Structure: Generate all code within a single HTML file for simplicity "
        "- No External Libraries: Do not use external JavaScript libraries unless specifically requested "
        ""
        "Always ask clarifying questions if the user's request is ambiguous, provide explanations for "
        "your code and design choices, and be prepared to iterate based on feedback."
    ),
    tools=[generate_html_component, validate_and_optimize_code],
)
