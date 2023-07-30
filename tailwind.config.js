/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./templates/*.{html,js}"],
    theme: {
        extend: {
            animation: {
                "astronaut-spin": "spin 3s linear infinite"
            }
        }
    },
    plugins: [require("daisyui")]
};
