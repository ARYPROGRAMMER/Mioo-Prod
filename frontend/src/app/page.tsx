import ChatInterface from "./components/ChatInterface";

export default function Home() {
  const userId = "user-" + Math.random().toString(36).substr(2, 9);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 via-indigo-50/30 to-violet-50/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-10 max-w-[1800px] h-screen flex flex-col">
        <header className="mb-4 md:mb-6 text-center">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight mb-2">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-600 to-indigo-600">
              Mioo
            </span>
            <span className="text-gray-800"> AI Tutor</span>
          </h1>
          <p className="text-gray-500 text-sm md:text-base max-w-2xl mx-auto">
            Personalized learning powered by reinforcement learning and advanced
            AI
          </p>
        </header>

        <div className="flex-1 overflow-hidden pb-4 h-[calc(100vh-160px)]">
          <ChatInterface userId={userId} />
        </div>

        <footer className="mt-2 pt-2 border-t border-gray-200">
          <div className="flex flex-col md:flex-row justify-between items-center text-xs text-gray-400">
            <div className="mb-1 md:mb-0">
              © {new Date().getFullYear()} Mioo AI Tutor | All rights reserved
            </div>
            <div className="flex space-x-4">
              <span>Privacy Policy</span>
              <span>Terms of Service</span>
              <span>Help</span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
