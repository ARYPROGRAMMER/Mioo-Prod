import ChatInterface from "./components/ChatInterface";

export default function Home() {
  const userId = "user-" + Math.random().toString(36).substr(2, 9);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-gray-100 to-indigo-50 py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-800 mb-8 text-center">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600">
              Mioo - Your Personal AI Tutor
            </span>
          </h1>
          <ChatInterface userId={userId} />
        </div>
      </div>
    </main>
  );
}
