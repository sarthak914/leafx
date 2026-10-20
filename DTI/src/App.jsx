import { Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Navbar from "./layers/ui/components/Navbar";
import BottomNav from "./layers/ui/components/BottomNav";
import PageTransition from "./layers/ui/components/PageTransition";
import Landing from "./layers/ui/pages/Landing";
import Scan from "./layers/ui/pages/Scan";
import Results from "./layers/ui/pages/Results";
import History from "./layers/ui/pages/History";

function App() {
  return (
    <div className="relative min-h-screen overflow-x-hidden bg-brand text-textSoft">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_14%_14%,rgba(0,255,156,0.1),transparent_36%),radial-gradient(circle_at_85%_16%,rgba(0,207,255,0.1),transparent_40%)]" />
      <div className="pointer-events-none fixed inset-0 mesh-bg opacity-15" />

      <Navbar />
      <AnimatePresence mode="wait">
        <PageTransition>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/scan" element={<Scan />} />
            <Route path="/results" element={<Results />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </PageTransition>
      </AnimatePresence>
      <BottomNav />
    </div>
  );
}

export default App;
