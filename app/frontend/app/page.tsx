"use client";
import { useState, useEffect } from "react";
import axios from "axios";
import { Shield, Zap, Brain, Bot, Plus, Play, Save, Upload, AlertTriangle, CheckCircle, XCircle, Trash2, ArrowUp, ArrowDown, Settings, Pencil } from "lucide-react";

// --- TYPES ---
type Guard = {
  id: number;
  name: string;
  enable_heuristics: boolean;
  enable_classifier: boolean;
  enable_llm: boolean;
  layer_order: string;
  heuristic_threshold?: number;
  confidence_threshold?: number;
  classifier_path?: string;
  llm_verifier?: string;
  llm_prompt?: string;
  hf_token?: string;
  created_at?: string;
};

// --- API CLIENT ---
const api = axios.create({ baseURL: "http://localhost:8000" });

// --- BUILD MODE STATE TYPE ---
type BuildModeState = {
  activeGuard: Guard | null;
  showBuilder: boolean;
  editingGuard: any;
  // InteractiveTester state (preserved when switching modes)
  testerPrompt: string;
  testerResult: any;
  testerHfToken: string;
  testerShowOptions: boolean;
};

// --- EVALUATION MODE STATE TYPE ---
type EvaluationModeState = {
  selectedGuard: number | null;
  selectedGuard2: number | null;
  results: any;
  compareResults: any;
  hfToken: string;
  showTokenInput: boolean;
  compareMode: boolean;
  expandedCard: string | null;
};

// --- COMPONENT: BUILD MODE ---
function BuildMode({ 
  guards, 
  refreshGuards,
  state,
  setState
}: { 
  guards: Guard[]; 
  refreshGuards: () => void;
  state: BuildModeState;
  setState: React.Dispatch<React.SetStateAction<BuildModeState>>;
}) {
  const { activeGuard, showBuilder, editingGuard } = state;
  
  const setActiveGuard = (guard: Guard | null) => setState(s => ({ ...s, activeGuard: guard }));
  const setShowBuilder = (show: boolean) => setState(s => ({ ...s, showBuilder: show }));
  const setEditingGuard = (guard: any) => setState(s => ({ ...s, editingGuard: guard }));

  // Edit Guard
  const editGuard = async (id: number) => {
    try {
      const res = await api.get(`/guards/${id}`);
      setEditingGuard(res.data);
      setShowBuilder(true);
    } catch (error) {
      console.error("Failed to load guardrail:", error);
    }
  };

  // Delete Guard
  const deleteGuard = (id: number) => {
    const confirmed = window.confirm("Are you sure you want to delete this guardrail?");
    if (!confirmed) return;
    
    api.delete(`/guards/${id}`)
      .then(() => {
        refreshGuards();
        if (activeGuard?.id === id) setActiveGuard(null);
      })
      .catch((error) => {
        console.error("Failed to delete guardrail:", error);
      });
  };

  // Reorder Guards
  const moveGuard = async (index: number, direction: "up" | "down") => {
    const newGuards = [...guards];
    const targetIndex = direction === "up" ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= newGuards.length) return;
    
    [newGuards[index], newGuards[targetIndex]] = [newGuards[targetIndex], newGuards[index]];
    refreshGuards();
  };

  // Delete All Guards
  const deleteAllGuards = async () => {
    if (!confirm("⚠️ DELETE ALL GUARDRAILS? This cannot be undone!")) return;
    
    try {
      await api.delete("/guards");
      refreshGuards();
      setActiveGuard(null);
    } catch (error) {
      console.error("Failed to delete all:", error);
    }
  };

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Header Card */}
      <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="relative flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              Build Mode
            </h2>
            <p className="text-gray-400">Create and test your guardrails interactively</p>
          </div>
          <button 
            onClick={() => setShowBuilder(true)}
            className="group relative px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 font-medium transition-all shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 active:scale-95"
          >
            <div className="absolute inset-0 bg-white/20 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center gap-2">
              <Plus className="w-4 h-4" />
              <span>New Guardrail</span>
            </div>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SIDEBAR: ACTIVE GUARDS */}
        <aside className="lg:col-span-4 xl:col-span-3 space-y-4">
          <div className="flex justify-between items-center px-1">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Your Guardrails</h2>
            {guards.length > 0 && (
              <button 
                onClick={deleteAllGuards}
                className="text-xs text-red-400/70 hover:text-red-300 transition-colors hover:scale-105 active:scale-95"
                title="Delete all guardrails"
              >
                Clear All
              </button>
            )}
          </div>
          <p className="text-xs text-gray-500 px-1 -mt-2">Configure and test your defense layers</p>
          <div className="space-y-3">
            {guards.map((g, idx) => (
              <div 
                key={g.id}
                onClick={() => setActiveGuard(g)}
                className={`group relative p-4 rounded-2xl border cursor-pointer transition-all backdrop-blur-sm overflow-hidden ${
                  activeGuard?.id === g.id 
                    ? "bg-blue-500/10 border-blue-500/50 shadow-lg shadow-blue-500/20" 
                    : "bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10"
                }`}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/10 to-blue-500/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                
                <div className="absolute top-3 right-3 flex gap-1 opacity-0 group-hover:opacity-100 transition-all z-10">
                  {idx > 0 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); moveGuard(idx, "up"); }}
                      className="p-1.5 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg transition-all hover:scale-110 active:scale-95"
                      title="Move up"
                    >
                      <ArrowUp className="w-3 h-3" />
                    </button>
                  )}
                  {idx < guards.length - 1 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); moveGuard(idx, "down"); }}
                      className="p-1.5 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg transition-all hover:scale-110 active:scale-95"
                      title="Move down"
                    >
                      <ArrowDown className="w-3 h-3" />
                    </button>
                  )}
                  <button
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      editGuard(g.id);
                    }}
                    className="p-1.5 bg-blue-500/20 hover:bg-blue-500/30 backdrop-blur-sm rounded-lg transition-all hover:scale-110 active:scale-95"
                    title="Edit guardrail"
                  >
                    <Pencil className="w-3 h-3" />
                  </button>
                  <button
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteGuard(g.id);
                    }}
                    className="p-1.5 bg-red-500/20 hover:bg-red-500/30 backdrop-blur-sm rounded-lg transition-all hover:scale-110 active:scale-95"
                    title="Delete guardrail"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
                
                <div className="relative font-bold text-base pr-20 mb-2">{g.name}</div>
                <div className="relative flex flex-wrap gap-2 text-xs">
                  {!!g.enable_heuristics && (
                    <span className="px-2.5 py-1 rounded-lg bg-yellow-500/10 text-yellow-300 border border-yellow-500/20 backdrop-blur-sm">
                      ⚡ Fast
                    </span>
                  )}
                  {!!g.enable_classifier && (
                    <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-300 border border-purple-500/20 backdrop-blur-sm">
                      🧠 Neural
                    </span>
                  )}
                  {!!g.enable_llm && (
                    <span className="px-2.5 py-1 rounded-lg bg-green-500/10 text-green-300 border border-green-500/20 backdrop-blur-sm">
                      🤖 LLM
                    </span>
                  )}
                </div>
              </div>
            ))}
            {guards.length === 0 && (
              <div className="text-center py-12 text-gray-500 italic border-2 border-dashed border-white/10 rounded-2xl backdrop-blur-sm">
                <Shield className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p>No guardrails yet</p>
                <p className="text-xs mt-1">Create one to get started</p>
              </div>
            )}
          </div>
        </aside>

        {/* MAIN: INTERACTIVE TESTER */}
        <section className="lg:col-span-8 xl:col-span-9">
          {activeGuard ? (
            <InteractiveTester 
              guard={activeGuard}
              prompt={state.testerPrompt}
              setPrompt={(p: string) => setState(s => ({ ...s, testerPrompt: p }))}
              result={state.testerResult}
              setResult={(r: any) => setState(s => ({ ...s, testerResult: r }))}
              hfToken={state.testerHfToken}
              setHfToken={(t: string) => setState(s => ({ ...s, testerHfToken: t }))}
              showOptions={state.testerShowOptions}
              setShowOptions={(show: boolean) => setState(s => ({ ...s, testerShowOptions: show }))}
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 border-2 border-dashed border-white/10 rounded-2xl p-16 backdrop-blur-sm bg-white/5 min-h-[500px]">
              <div className="relative">
                <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-2xl animate-pulse-slow"></div>
                <Shield className="relative w-20 h-20 mb-4 opacity-30" />
              </div>
              <p className="text-lg font-medium mb-2">Select a Guardrail</p>
              <p className="text-sm text-gray-500">Choose from the sidebar or create a new one to start testing</p>
            </div>
          )}
        </section>
      </div>

      {/* POPUP: ARCHITECTURE BUILDER */}
      {showBuilder && (
        <ArchitectureBuilder 
          onClose={() => { setShowBuilder(false); setEditingGuard(null); }} 
          onSave={refreshGuards} 
          editingGuard={editingGuard}
        />
      )}
    </div>
  );
}

// --- COMPONENT: INTERACTIVE TESTER ---
function InteractiveTester({ 
  guard,
  prompt,
  setPrompt,
  result,
  setResult,
  hfToken,
  setHfToken,
  showOptions,
  setShowOptions
}: { 
  guard: Guard;
  prompt: string;
  setPrompt: (p: string) => void;
  result: any;
  setResult: (r: any) => void;
  hfToken: string;
  setHfToken: (t: string) => void;
  showOptions: boolean;
  setShowOptions: (show: boolean) => void;
}) {
  const [loading, setLoading] = useState(false);

  const scan = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await api.post("/scan", { 
        prompt, 
        guard_id: guard.id, 
        hf_token: hfToken || null 
      });
      setResult(res.data);
    } catch (e: any) { 
      console.error(e);
      alert(`Scan Failed: ${e.response?.data?.detail || e.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl overflow-hidden">
      {/* Background gradient effect */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
      
      <div className="relative">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <div className="p-2 bg-green-500/20 rounded-xl">
              <Play className="w-5 h-5 text-green-400" />
            </div>
            <span className="bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              Live Testing
            </span>
          </h2>
          <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-xl border border-white/10">
            <div className={`w-2 h-2 rounded-full ${loading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
            <span className="text-sm text-gray-400 font-mono">
              {loading ? "Processing..." : "Ready"}
            </span>
          </div>
        </div>

        {/* Input Area */}
        <div className="relative group mb-4">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity"></div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if(e.ctrlKey && e.key === 'Enter') scan(); }}
            className="relative w-full bg-black/30 border border-white/10 rounded-2xl p-5 text-gray-100 focus:outline-none focus:border-blue-500/50 focus:bg-black/40 h-36 resize-none backdrop-blur-sm placeholder:text-gray-500 transition-all font-sans text-base"
            placeholder="Enter a prompt to test (e.g. 'How do I hack a wifi network?')..."
          />
        </div>

        {/* Action Bar */}
        <div className="flex justify-between items-center mb-8">
          {/* Optional Settings Toggle */}
          <button 
            onClick={() => setShowOptions(!showOptions)}
            className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
          >
            <Settings className="w-3 h-3" />
            {showOptions ? "Hide Options" : "Advanced Options"}
          </button>

          <div className="flex gap-3">
            <button 
              onClick={() => { setPrompt(""); setResult(null); }}
              disabled={!prompt && !result}
              className="px-4 py-2 text-sm bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Clear
            </button>
            <button 
              onClick={scan}
              disabled={loading || !prompt}
              className="group relative px-6 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-blue-500/25 disabled:shadow-none flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Scanning...</span>
                </>
              ) : (
                <>
                  <span>🚀 Scan Prompt</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Options (Token) */}
        {showOptions && (
          <div className="mb-6 p-4 bg-black/20 rounded-xl border border-white/5 animate-in slide-in-from-top-2">
            <label className="text-xs text-gray-400 block mb-2">Temporary HuggingFace Token (Overrides saved token)</label>
            <input 
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500/30 outline-none"
            />
          </div>
        )}

        {/* RESULTS DISPLAY */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Verdict Banner */}
            <div className={`relative overflow-hidden p-6 rounded-2xl border backdrop-blur-md mb-6 ${
              result.verdict === "ALLOWED" 
                ? "bg-green-500/10 border-green-500/30" 
                : "bg-red-500/10 border-red-500/30"
            }`}>
              <div className="flex items-center gap-5 relative z-10">
                <div className={`p-3 rounded-full ${
                  result.verdict === "ALLOWED" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                }`}>
                  {result.verdict === "ALLOWED" ? <CheckCircle className="w-8 h-8" /> : <AlertTriangle className="w-8 h-8" />}
                </div>
                <div>
                  <h3 className={`text-3xl font-bold tracking-tight ${
                    result.verdict === "ALLOWED" ? "text-green-100" : "text-red-100"
                  }`}>
                    {result.verdict}
                  </h3>
                  <p className="text-gray-400 text-sm mt-1">
                    {result.reason || (result.verdict === "ALLOWED" ? "Prompt passed all security checks." : "Prompt blocked by security layers.")}
                  </p>
                </div>
              </div>
            </div>

            {/* Detailed Steps */}
            <div className="bg-black/20 rounded-2xl p-6 border border-white/5">
              <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Pipeline Analysis</h4>
              <div className="space-y-3">
                {result.steps.map((step: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-gray-500">{idx + 1}</span>
                      <span className="text-sm font-medium text-gray-300">{step.layer}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 font-mono max-w-[200px] truncate hidden md:block">
                        {step.detail}
                      </span>
                      <span className={`text-xs font-bold px-2 py-1 rounded-md ${
                        step.status === "PASS" ? "bg-green-500/20 text-green-400" :
                        step.status === "BLOCK" ? "bg-red-500/20 text-red-400" :
                        "bg-gray-500/20 text-gray-400"
                      }`}>
                        {step.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --- COMPONENT: ARCHITECTURE BUILDER (POPUP) ---
function ArchitectureBuilder({ onClose, onSave, editingGuard }: { onClose: () => void; onSave: () => void; editingGuard?: any }) {
  const [name, setName] = useState(editingGuard?.name || "");
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [heuristicThreshold, setHeuristicThreshold] = useState(editingGuard?.heuristic_threshold || 20);
  const [enableHeuristics, setEnableHeuristics] = useState(editingGuard?.enable_heuristics ?? true);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  
  // Initialize layers from editingGuard or defaults
  const getInitialLayers = () => {
    if (editingGuard) {
      const layers = [];
      const isClassifierFirst = editingGuard.layer_order === "classifier_first";
      
      if (editingGuard.enable_classifier) {
        layers.push({
          id: Date.now(),
          type: "classifier" as const,
          name: "Neural Classifier",
          icon: "🧠",
          color: "purple",
          enabled: true,
          file: null as File | null,
          folderPath: editingGuard.classifier_path || "",
          hfToken: editingGuard.hf_token || "",
          llmModel: "",
          confidenceThreshold: editingGuard.confidence_threshold || 0.7,
          llmPrompt: ""
        });
      }
      
      if (editingGuard.enable_llm) {
        layers.push({
          id: Date.now() + 1,
          type: "llm" as const,
          name: "LLM Verifier",
          icon: "🤖",
          color: "green",
          enabled: true,
          file: null as File | null,
          folderPath: "",
          hfToken: "",
          llmModel: editingGuard.llm_verifier || "",
          confidenceThreshold: 0.7,
          llmPrompt: editingGuard.llm_prompt || ""
        });
      }
      
      // Reorder based on layer_order
      if (!isClassifierFirst && layers.length === 2) {
        return [layers[1], layers[0]];
      }
      
      return layers.length > 0 ? layers : [
        { id: Date.now(), type: "classifier" as const, name: "Neural Classifier", icon: "🧠", color: "purple", enabled: true, file: null as File | null, folderPath: "", hfToken: "", llmModel: "", confidenceThreshold: 0.7, llmPrompt: "" }
      ];
    }
    
    return [
      { id: Date.now(), type: "classifier" as const, name: "Neural Classifier", icon: "🧠", color: "purple", enabled: true, file: null as File | null, folderPath: "", hfToken: "", llmModel: "", confidenceThreshold: 0.7, llmPrompt: "" },
      { id: Date.now() + 1, type: "llm" as const, name: "LLM Verifier", icon: "🤖", color: "green", enabled: true, file: null as File | null, folderPath: "", hfToken: "", llmModel: "", confidenceThreshold: 0.7, llmPrompt: "" }
    ];
  };
  
  const [layers, setLayers] = useState(getInitialLayers());

  useEffect(() => {
    api.get("/ollama-models").then(res => setOllamaModels(res.data));
  }, []);

  const addLayer = () => {
    if (layers.length >= 3) {
      alert("Maximum 3 layers allowed");
      return;
    }
    const newLayer = {
      id: Date.now(),
      type: "classifier" as "classifier" | "llm",
      name: "Neural Classifier",
      icon: "🧠",
      color: "purple",
      enabled: true,
      file: null as File | null,
      folderPath: "",
      hfToken: "",
      llmModel: "",
      confidenceThreshold: 0.7,
      llmPrompt: ""
    };
    setLayers([...layers, newLayer]);
  };

  const removeLayer = (id: number) => {
    if (layers.length <= 1) {
      alert("Must have at least one layer");
      return;
    }
    setLayers(layers.filter(l => l.id !== id));
  };

  const changeLayerType = (id: number, type: "classifier" | "llm") => {
    setLayers(layers.map(l => l.id === id ? {
      ...l,
      type,
      name: type === "classifier" ? "Neural Classifier" : "LLM Verifier",
      icon: type === "classifier" ? "🧠" : "🤖",
      color: type === "classifier" ? "purple" : "green",
      folderPath: "",
      hfToken: "",
      file: null,
      llmModel: "",
      confidenceThreshold: 0.7,
      llmPrompt: ""
    } : l));
  };

  const updateLayerConfidence = (id: number, threshold: number) => {
    setLayers(layers.map(l => l.id === id ? {...l, confidenceThreshold: threshold} : l));
  };

  const updateLayerPrompt = (id: number, prompt: string) => {
    setLayers(layers.map(l => l.id === id ? {...l, llmPrompt: prompt} : l));
  };

  const updateLayerFile = (id: number, file: File | null) => {
    setLayers(layers.map(l => l.id === id ? {...l, file} : l));
  };

  const updateLayerFolderPath = (id: number, path: string) => {
    setLayers(layers.map(l => l.id === id ? {...l, folderPath: path, file: null} : l));
  };

  const updateLayerHfToken = (id: number, token: string) => {
    setLayers(layers.map(l => l.id === id ? {...l, hfToken: token} : l));
  };

  const handleFolderSelect = (id: number, files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    // Extract folder path from the first file's path
    const firstFile = files[0];
    // @ts-ignore - webkitRelativePath exists but TypeScript doesn't know
    const fullPath = firstFile.webkitRelativePath || firstFile.name;
    
    // Try to extract absolute path (browser limitations mean this often only gives folder name)
    // For best results, users should paste the absolute path in the text field
    let folderPath = fullPath.split('/')[0];
    
    // Show warning that absolute path is needed
    console.warn('Browser file picker returned:', folderPath, '- Paste absolute path in text field for local models');
    
    updateLayerFolderPath(id, folderPath);
  };

  const updateLayerLLM = (id: number, llmModel: string) => {
    setLayers(layers.map(l => l.id === id ? {...l, llmModel} : l));
  };

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = "move";
    const ghost = document.createElement('div');
    ghost.style.opacity = '0';
    document.body.appendChild(ghost);
    e.dataTransfer.setDragImage(ghost, 0, 0);
    setTimeout(() => document.body.removeChild(ghost), 0);
  };

  const handleDragEnter = (index: number) => {
    if (draggedIndex !== null && draggedIndex !== index) {
      setDragOverIndex(index);
    }
  };

  const handleDragEnd = () => {
    if (draggedIndex !== null && dragOverIndex !== null) {
      const newLayers = [...layers];
      const [draggedLayer] = newLayers.splice(draggedIndex, 1);
      newLayers.splice(dragOverIndex, 0, draggedLayer);
      setLayers(newLayers);
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const toggleLayer = (id: number) => {
    setLayers(layers.map(l => l.id === id ? {...l, enabled: !l.enabled} : l));
  };

  const handleSave = async () => {
    // Use folder paths directly (no upload needed for local paths)
    const layersWithPaths = layers.map((layer) => {
      if (layer.type === "classifier" && layer.folderPath) {
        return { ...layer, classifier_path: layer.folderPath };
      }
      return { ...layer, classifier_path: "" };
    });

    // For now, save only the first enabled classifier and LLM (backend limitation)
    const firstClassifier = layersWithPaths.find(l => l.type === "classifier" && l.enabled);
    const firstLLM = layersWithPaths.find(l => l.type === "llm" && l.enabled);
    
    // Determine order
    const classifierIndex = layers.findIndex(l => l.type === "classifier" && l.enabled);
    const llmIndex = layers.findIndex(l => l.type === "llm" && l.enabled);
    let layer_order = "classifier_first";
    if (classifierIndex !== -1 && llmIndex !== -1) {
      layer_order = classifierIndex < llmIndex ? "classifier_first" : "llm_first";
    }

    const guardData = {
      name,
      classifier_path: firstClassifier?.classifier_path || "",
      heuristic_threshold: heuristicThreshold,
      llm_verifier: firstLLM?.llmModel || "",
      enable_heuristics: enableHeuristics,
      enable_classifier: !!firstClassifier,
      enable_llm: !!firstLLM,
      layer_order,
      confidence_threshold: firstClassifier?.confidenceThreshold || 0.7,
      llm_prompt: firstLLM?.llmPrompt || "",
      hf_token: firstClassifier?.hfToken || ""
    };

    // Update existing or create new
    if (editingGuard?.id) {
      await api.put(`/guards/${editingGuard.id}`, guardData);
    } else {
      await api.post("/save-guard", guardData);
    }
    
    onSave();
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <div className="relative bg-gradient-to-br from-gray-900/95 to-gray-950/95 backdrop-blur-2xl border border-white/10 w-full max-w-5xl rounded-3xl shadow-2xl overflow-hidden">
        {/* Decorative gradients */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
        
        <div className="relative p-8">
          {/* Close button */}
          <button 
            onClick={onClose} 
            className="absolute top-6 right-6 p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-xl transition-all hover:scale-110 active:scale-95"
          >
            <span className="text-2xl">✕</span>
          </button>
          
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 bg-blue-500/20 rounded-2xl">
                <Settings className="text-blue-400 w-7 h-7" />
              </div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                {editingGuard ? "Edit Guardrail" : "Build Custom Guardrail"}
              </h2>
            </div>
            <p className="text-gray-400 text-sm ml-16">{editingGuard ? "Modify your guardrail configuration" : "Configure and reorder your defense layers"}</p>
          </div>

          <div className="space-y-6 max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
            {/* NAME */}
            <div className="group">
              <label className="block text-sm font-semibold text-gray-300 mb-2 ml-1">Guardrail Name</label>
              <div className="relative">
                <input 
                  value={name} 
                  onChange={e => setName(e.target.value)} 
                  className="w-full bg-black/30 border border-white/10 rounded-xl px-5 py-3.5 focus:border-blue-500/50 focus:bg-black/40 outline-none text-white placeholder:text-gray-500 transition-all backdrop-blur-sm"
                  placeholder="e.g. Production Defense V1"
                />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity -z-10"></div>
              </div>
            </div>

            {/* HEURISTICS TOGGLE */}
            <div className="relative bg-yellow-500/5 backdrop-blur-sm border border-yellow-500/20 rounded-2xl p-5 overflow-hidden group hover:border-yellow-500/30 transition-all">
              <div className="absolute top-0 right-0 w-32 h-32 bg-yellow-500/10 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              
              <div className="relative flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-yellow-500/20 rounded-xl">
                    <Zap className="w-5 h-5 text-yellow-400" />
                  </div>
                  <div>
                    <div className="font-bold text-yellow-100">Speed Heuristics</div>
                    <div className="text-xs text-gray-400 mt-0.5">Fast-path optimization for short prompts</div>
                  </div>
                </div>
                <button
                  onClick={() => setEnableHeuristics(!enableHeuristics)}
                  className={`px-5 py-2 rounded-xl font-medium text-sm transition-all hover:scale-105 active:scale-95 ${
                    enableHeuristics
                      ? "bg-yellow-500/20 border-2 border-yellow-500/50 text-yellow-300 shadow-lg shadow-yellow-500/20"
                      : "bg-white/5 text-gray-400 border-2 border-white/10 hover:border-white/20"
                  }`}
                >
                  {enableHeuristics ? "Enabled" : "Disabled"}
                </button>
              </div>
              
              {enableHeuristics && (
                <div className="relative pt-4 border-t border-yellow-500/20 animate-slide-up">
                  <label className="text-xs text-gray-400 block mb-2 ml-1">Short Prompt Threshold (characters)</label>
                  <input 
                    type="number"
                    value={heuristicThreshold}
                    onChange={e => setHeuristicThreshold(parseInt(e.target.value) || 20)}
                    className="w-full bg-black/30 border border-yellow-500/20 rounded-xl px-4 py-2.5 text-sm text-white focus:border-yellow-500/50 outline-none backdrop-blur-sm"
                    placeholder="20"
                  />
                  <p className="text-xs text-gray-500 mt-2 ml-1">Prompts shorter than this skip heavyweight processing</p>
                </div>
              )}
            </div>

            {/* HORIZONTAL DRAG & DROP LAYERS */}
            <div>
              <div className="flex justify-between items-center mb-3 px-1">
                <label className="block text-sm font-semibold text-gray-300">
                  Detection Layers <span className="text-gray-500 font-normal text-xs">(Drag to reorder • Max 3)</span>
                </label>
                <button
                  onClick={addLayer}
                  disabled={layers.length >= 3}
                  className="text-xs bg-blue-600/80 hover:bg-blue-500 disabled:opacity-30 disabled:cursor-not-allowed text-white px-4 py-2 rounded-xl flex items-center gap-1.5 font-medium transition-all hover:scale-105 active:scale-95 shadow-lg shadow-blue-500/20 disabled:shadow-none"
                >
                  <Plus className="w-3.5 h-3.5" /> Add Layer
                </button>
              </div>
            </div>
            
            <div className="flex gap-4 p-5 bg-black/20 backdrop-blur-sm rounded-2xl border border-white/10 justify-center overflow-x-auto">
              {layers.map((layer, index) => {
                const colorMap: any = {
                  purple: { bg: "bg-purple-500/10", border: "border-purple-500/50", text: "text-purple-300", shadow: "shadow-purple-500/20", glow: "from-purple-500/20" },
                  green: { bg: "bg-green-500/10", border: "border-green-500/50", text: "text-green-300", shadow: "shadow-green-500/20", glow: "from-green-500/20" }
                };
                const colors = colorMap[layer.color];
                
                return (
                  <div
                    key={layer.id}
                    className={`relative flex-shrink-0 w-[240px] p-5 border-2 rounded-2xl transition-all backdrop-blur-sm overflow-hidden group ${
                      layer.enabled 
                        ? `${colors.bg} ${colors.border} ${colors.shadow} shadow-xl hover:scale-105` 
                        : "bg-white/5 border-white/10 opacity-60 hover:opacity-80"
                    } ${draggedIndex === index ? 'scale-110 rotate-3 opacity-50 z-50' : ''} ${dragOverIndex === index && draggedIndex !== index ? 'scale-95 -rotate-1' : ''}`}
                    onDragEnter={() => handleDragEnter(index)}
                    onDragEnd={handleDragEnd}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    {/* Gradient glow effect */}
                    {layer.enabled && (
                      <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${colors.glow} to-transparent rounded-full blur-2xl opacity-50`}></div>
                    )}
                    
                    {/* Layer Type Selector */}
                    <div className="relative mb-3 space-y-2">
                      <select
                        value={layer.type}
                        onChange={(e) => changeLayerType(layer.id, e.target.value as "classifier" | "llm")}
                        className="w-full bg-black/40 border border-white/20 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-white/40 backdrop-blur-sm cursor-pointer hover:bg-black/50 transition-all"
                      >
                        <option value="classifier">🧠 Neural Classifier</option>
                        <option value="llm">🤖 LLM Verifier</option>
                      </select>
                      
                      {layers.length > 1 && (
                        <button
                          onClick={() => removeLayer(layer.id)}
                          className="w-full text-xs text-red-400/80 hover:text-red-300 hover:bg-red-500/10 py-1.5 rounded-lg transition-all"
                        >
                          Remove Layer
                        </button>
                      )}
                    </div>
                    
                    {/* DRAG HANDLE */}
                    <div 
                      className="relative text-center mb-4 py-3 border-y border-white/10 cursor-move select-none hover:bg-white/5 rounded-lg transition-all"
                      draggable={layer.enabled}
                      onDragStart={(e) => handleDragStart(e, index)}
                    >
                      <div className="text-4xl mb-2 filter hover:scale-110 transition-transform">{layer.icon}</div>
                      <div className="font-bold text-sm mb-1">{layer.name}</div>
                      {layer.enabled && (
                        <div className="text-xs text-gray-500 flex items-center justify-center gap-1">
                          <span className="opacity-60">⋮⋮</span>
                          <span>Drag to reorder</span>
                        </div>
                      )}
                    </div>
                    
                    <button
                      onClick={() => toggleLayer(layer.id)}
                      className={`relative w-full px-4 py-2 rounded-xl font-medium text-xs transition-all mb-4 overflow-hidden group/btn ${
                        layer.enabled
                          ? `${colors.bg} ${colors.border} border-2 ${colors.text} hover:scale-105 shadow-lg`
                          : "bg-white/5 text-gray-400 border-2 border-white/10 hover:border-white/20"
                      }`}
                    >
                      <span className="relative z-10">{layer.enabled ? "✓ Enabled" : "Disabled"}</span>
                      {layer.enabled && (
                        <div className="absolute inset-0 bg-white/10 opacity-0 group-hover/btn:opacity-100 transition-opacity"></div>
                      )}
                    </button>
                    
                    {layer.enabled && layer.type === "classifier" && (
                      <div className="relative pt-4 border-t border-purple-500/20 space-y-4 animate-slide-up">
                        <div>
                          <label className="text-xs text-gray-400 block mb-2 ml-1">Model Folder</label>
                          <div className="flex gap-2">
                            <input 
                              type="text"
                              value={layer.folderPath}
                              onChange={(e) => updateLayerFolderPath(layer.id, e.target.value)}
                              placeholder="Select folder or paste path..."
                              className="flex-1 bg-black/40 border border-white/20 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-purple-500/50 backdrop-blur-sm placeholder:text-gray-600"
                            />
                            <label className="cursor-pointer px-3 py-2 bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 rounded-xl text-xs font-medium text-purple-300 transition-all flex items-center gap-1.5">
                              📁 Browse
                              <input 
                                type="file" 
                                // @ts-ignore - webkitdirectory is not in TypeScript types
                                webkitdirectory=""
                                directory=""
                                multiple
                                onChange={(e) => handleFolderSelect(layer.id, e.target.files)}
                                className="hidden"
                              />
                            </label>
                          </div>
                          {layer.folderPath && (
                            <p className="mt-2 text-xs text-green-400 flex items-center gap-1 ml-1">
                              <CheckCircle className="w-3 h-3" />
                              {layer.folderPath}
                            </p>
                          )}
                          <p className="text-xs text-amber-400/70 mt-2 ml-1 font-medium">⚠️ For local models: Paste the full absolute path (e.g., C:\Users\...\model_folder)</p>
                        </div>
                        <div>
                          <label className="text-xs text-gray-400 block mb-2 ml-1">HuggingFace Token (optional)</label>
                          <input 
                            type="password"
                            value={layer.hfToken}
                            onChange={(e) => updateLayerHfToken(layer.id, e.target.value)}
                            placeholder="hf_..."
                            className="w-full bg-black/40 border border-white/20 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-purple-500/50 backdrop-blur-sm placeholder:text-gray-600"
                          />
                          <p className="text-xs text-gray-500 mt-1 ml-1">Required for private HuggingFace models</p>
                        </div>
                        <div 
                          onMouseDown={e => e.stopPropagation()}
                          onDragStart={e => e.preventDefault()}
                          draggable={false}
                        >
                          <label className="text-xs text-gray-400 block mb-2 ml-1">Confidence Threshold</label>
                          <input 
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={layer.confidenceThreshold}
                            onChange={e => updateLayerConfidence(layer.id, parseFloat(e.target.value))}
                            className="w-full cursor-pointer accent-purple-500"
                          />
                          <div className="text-sm text-center text-purple-300 font-mono font-bold mt-1">{layer.confidenceThreshold.toFixed(2)}</div>
                          <p className="text-xs text-gray-500 mt-1 ml-1">Block if confidence ≥ threshold</p>
                        </div>
                      </div>
                    )}
                    
                    {layer.enabled && layer.type === "llm" && (
                      <div className="relative pt-4 border-t border-green-500/20 space-y-4 animate-slide-up">
                        <div>
                          <label className="text-xs text-gray-400 block mb-2 ml-1">Ollama Model</label>
                          <select 
                            value={layer.llmModel}
                            className="w-full bg-black/40 border border-white/20 rounded-xl px-3 py-2 text-xs text-white outline-none focus:border-green-500/50 backdrop-blur-sm"
                            onChange={e => updateLayerLLM(layer.id, e.target.value)}
                          >
                            <option value="">Select model...</option>
                            {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
                          </select>
                          {ollamaModels.length === 0 && (
                            <p className="text-xs text-red-400 mt-2 ml-1 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" />
                              No Ollama models detected
                            </p>
                          )}
                        </div>
                        <div>
                          <label className="text-xs text-gray-400 block mb-2 ml-1">System Prompt (optional)</label>
                          <textarea
                            value={layer.llmPrompt}
                            onChange={e => updateLayerPrompt(layer.id, e.target.value)}
                            placeholder="Custom instructions for the LLM..."
                            className="w-full bg-black/40 border border-white/20 rounded-xl px-3 py-2 text-xs text-white min-h-[70px] resize-none outline-none focus:border-green-500/50 backdrop-blur-sm placeholder:text-gray-600"
                          />
                          <p className="text-xs text-gray-500 mt-1 ml-1">Default: Classify as Benign/Harmful/Injection</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-gray-500 mt-3 ml-1 flex items-center gap-2">
              <span className="text-blue-400">💡</span>
              Layers execute left to right in the order shown above
            </p>
          </div>
        </div>

        <div className="relative mt-8 pt-6 border-t border-white/10 flex justify-between items-center">
          <p className="text-xs text-gray-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-400" />
            Enable at least one detection layer
          </p>
          <button 
            onClick={handleSave}
            disabled={!name || layers.every(l => !l.enabled)}
            className="group relative px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:opacity-40 disabled:cursor-not-allowed disabled:from-gray-700 disabled:to-gray-700 text-white rounded-xl font-bold transition-all hover:scale-105 active:scale-95 shadow-xl shadow-blue-500/30 disabled:shadow-none overflow-hidden"
          >
            <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative flex items-center gap-2">
              <Save className="w-5 h-5" />
              <span>{editingGuard ? "Update Guardrail" : "Save Guardrail"}</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}

// --- COMPONENT: EVALUATION MODE ---
function EvaluationMode({ 
  guards,
  state,
  setState
}: { 
  guards: Guard[];
  state: EvaluationModeState;
  setState: React.Dispatch<React.SetStateAction<EvaluationModeState>>;
}) {
  const { selectedGuard, selectedGuard2, results, compareResults, hfToken, showTokenInput, compareMode, expandedCard } = state;
  const [evaluating, setEvaluating] = useState(false);
  
  const setSelectedGuard = (id: number | null) => setState(s => ({ ...s, selectedGuard: id }));
  const setSelectedGuard2 = (id: number | null) => setState(s => ({ ...s, selectedGuard2: id }));
  const setResults = (r: any) => setState(s => ({ ...s, results: r }));
  const setCompareResults = (r: any) => setState(s => ({ ...s, compareResults: r }));
  const setHfToken = (t: string) => setState(s => ({ ...s, hfToken: t }));
  const setShowTokenInput = (show: boolean) => setState(s => ({ ...s, showTokenInput: show }));
  const setCompareMode = (mode: boolean) => setState(s => ({ ...s, compareMode: mode }));
  const setExpandedCard = (card: string | null) => setState(s => ({ ...s, expandedCard: card }));

  const evaluationSets = [
    {
      id: 1,
      name: "notInject",
      description: "Dataset for testing non-injection scenarios and benign inputs",
      hfDataset: "leolee99/NotInject",
      count: "339",
      color: "from-emerald-500/20 to-green-500/20",
      borderColor: "border-emerald-500/30 hover:border-emerald-400/50",
      available: true
    },
    {
      id: 2,
      name: "sqlInjection",
      description: "SQL injection attack patterns and variations",
      hfDataset: "Coming soon",
      count: "TBD",
      color: "from-red-500/20 to-orange-500/20",
      borderColor: "border-red-500/30 hover:border-red-400/50",
      available: false
    },
    {
      id: 3,
      name: "xssAttacks",
      description: "Cross-site scripting (XSS) test cases",
      hfDataset: "Coming soon",
      count: "TBD",
      color: "from-orange-500/20 to-yellow-500/20",
      borderColor: "border-orange-500/30 hover:border-orange-400/50",
      available: false
    },
    {
      id: 4,
      name: "commandInjection",
      description: "Command injection and system-level attacks",
      hfDataset: "Coming soon",
      count: "TBD",
      color: "from-purple-500/20 to-pink-500/20",
      borderColor: "border-purple-500/30 hover:border-purple-400/50",
      available: false
    }
  ];

  const runEvaluation = async (datasetName: string) => {
    if (!selectedGuard) {
      alert("Please select a guardrail first!");
      return;
    }

    if (compareMode && !selectedGuard2) {
      alert("Please select a second guardrail to compare!");
      return;
    }

    if (compareMode && selectedGuard === selectedGuard2) {
      alert("Please select two different guardrails to compare!");
      return;
    }

    if (!hfToken.trim()) {
      alert("Please enter your HuggingFace token. The NotInject dataset is private and requires authentication.");
      setShowTokenInput(true);
      return;
    }

    setEvaluating(true);
    setResults(null);
    setCompareResults(null);
    
    try {
      if (compareMode && selectedGuard2) {
        // Compare two guardrails
        console.log("[COMPARE] Sending compare request:", {
          guard_id_1: selectedGuard,
          guard_id_2: selectedGuard2,
          dataset_name: datasetName,
        });
        
        const res = await api.post("/compare", {
          guard_id_1: selectedGuard,
          guard_id_2: selectedGuard2,
          dataset_name: datasetName,
          max_samples: 1000,
          hf_token: hfToken
        });
        setCompareResults(res.data);
      } else {
        // Single guardrail evaluation
        console.log("[EVAL] Sending request:", {
          guard_id: selectedGuard,
          dataset_name: datasetName,
          max_samples: 100,
          hf_token: hfToken ? "***provided***" : "missing"
        });
        
        const res = await api.post("/evaluate", {
          guard_id: selectedGuard,
          dataset_name: datasetName,
          max_samples: 1000,
          hf_token: hfToken
        });
        setResults(res.data);
      }
    } catch (error: any) {
      console.error("Evaluation failed:", error);
      console.error("Error response:", error.response?.data);
      const errorMsg = error.response?.data?.detail || error.message;
      if (errorMsg.includes("401") || errorMsg.includes("authentication")) {
        alert("Authentication failed. Please check your HuggingFace token.");
        setShowTokenInput(true);
      } else {
        alert(`Evaluation failed: ${errorMsg}`);
      }
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Header Card */}
      <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="relative">
          <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
            Evaluation Mode
          </h2>
          <p className="text-gray-400">Run your guardrails against predefined evaluation datasets to measure performance and accuracy</p>
        </div>
      </div>

      {/* Guardrail Selector */}
      {guards.length > 0 && (
        <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Shield className="w-5 h-5 text-blue-400" />
              {compareMode ? "Select Two Guardrails to Compare" : "Select Guardrail to Evaluate"}
            </h3>
            <button
              onClick={() => {
                setCompareMode(!compareMode);
                setSelectedGuard2(null);
                setResults(null);
                setCompareResults(null);
              }}
              className={`px-4 py-2 rounded-xl font-medium text-sm transition-all hover:scale-105 active:scale-95 ${
                compareMode
                  ? "bg-purple-500/20 border-2 border-purple-500/50 text-purple-300"
                  : "bg-white/5 text-gray-400 border-2 border-white/10 hover:border-white/20"
              }`}
            >
              {compareMode ? "✓ Compare Mode" : "🔀 Compare Mode"}
            </button>
          </div>
          
          {compareMode && (
            <div className="mb-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-xl">
              <p className="text-sm text-purple-300">
                Select two guardrails below to compare their performance on the same dataset.
              </p>
            </div>
          )}
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {guards.map((guard) => {
              const isSelected1 = selectedGuard === guard.id;
              const isSelected2 = selectedGuard2 === guard.id;
              const isSelected = isSelected1 || isSelected2;
              
              return (
                <button
                  key={guard.id}
                  onClick={() => {
                    if (compareMode) {
                      if (isSelected1) {
                        setSelectedGuard(null);
                      } else if (isSelected2) {
                        setSelectedGuard2(null);
                      } else if (!selectedGuard) {
                        setSelectedGuard(guard.id);
                      } else if (!selectedGuard2) {
                        setSelectedGuard2(guard.id);
                      } else {
                        // Both selected, replace first
                        setSelectedGuard(guard.id);
                      }
                    } else {
                      setSelectedGuard(guard.id);
                    }
                  }}
                  className={`bg-white/5 backdrop-blur-sm border rounded-xl p-4 transition-all text-left relative ${
                    isSelected1
                      ? "border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/20"
                      : isSelected2
                      ? "border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/20"
                      : "border-white/10 hover:border-white/20 hover:bg-white/10"
                  }`}
                >
                  {compareMode && isSelected && (
                    <div className={`absolute -top-2 -right-2 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      isSelected1 ? "bg-blue-500 text-white" : "bg-purple-500 text-white"
                    }`}>
                      {isSelected1 ? "1" : "2"}
                    </div>
                  )}
                  <div className={`font-semibold text-sm mb-2 transition-colors ${
                    isSelected1 ? "text-blue-300" : isSelected2 ? "text-purple-300" : "text-white"
                  }`}>
                    {guard.name}
                  </div>
                  <div className="flex gap-1.5 text-xs flex-wrap">
                    {guard.enable_heuristics && (
                      <span className="bg-yellow-500/20 text-yellow-300 px-2 py-1 rounded-lg border border-yellow-500/30">⚡</span>
                    )}
                    {guard.enable_classifier && (
                      <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded-lg border border-purple-500/30">🧠</span>
                    )}
                    {guard.enable_llm && (
                      <span className="bg-green-500/20 text-green-300 px-2 py-1 rounded-lg border border-green-500/30">🤖</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {guards.length === 0 && (
        <div className="relative bg-yellow-500/5 backdrop-blur-sm border border-yellow-500/20 rounded-2xl p-6">
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-yellow-300 mb-2">No Guardrails Available</h4>
              <p className="text-sm text-gray-400">Create a guardrail first before running evaluations.</p>
            </div>
          </div>
        </div>
      )}

      {/* HuggingFace Token Input */}
      <div className="relative bg-blue-500/5 backdrop-blur-sm border border-blue-500/20 rounded-2xl p-6">
        <div className="flex items-start gap-4">
          <div className="p-2 bg-blue-500/20 rounded-xl flex-shrink-0">
            <Settings className="w-5 h-5 text-blue-400" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-blue-300">HuggingFace Authentication</h4>
              <button
                onClick={() => setShowTokenInput(!showTokenInput)}
                className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                {showTokenInput ? "Hide" : "Show"}
              </button>
            </div>
            <p className="text-sm text-gray-400 mb-3">
              The NotInject dataset is private and requires a HuggingFace token. 
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 ml-1 underline">
                Get your token here →
              </a>
            </p>
            
            {showTokenInput && (
              <div className="relative group animate-slide-up">
                <input
                  type="password"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="hf_..."
                  className="w-full bg-black/30 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:border-blue-500/50 outline-none backdrop-blur-sm placeholder:text-gray-600"
                />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity -z-10"></div>
                {hfToken && (
                  <div className="mt-2 text-xs text-green-400 flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" />
                    Token set (will be used for evaluation)
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Evaluation Sets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {evaluationSets.map((set, idx) => (
          <div
            key={set.id}
            className={`group relative bg-gradient-to-br ${set.color} backdrop-blur-sm border-2 ${set.borderColor} rounded-2xl p-6 transition-all hover:shadow-2xl overflow-hidden ${
              set.available ? "cursor-pointer hover:scale-105" : "opacity-60"
            }`}
            style={{ animationDelay: `${idx * 0.1}s` }}
          >
            <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            
            <div className="relative">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-bold text-white">{set.name}</h3>
                <div className="flex flex-col items-end gap-2">
                  <span className="px-3 py-1.5 bg-black/30 backdrop-blur-sm rounded-full text-xs font-mono border border-white/20">
                    {set.count} samples
                  </span>
                  {!set.available && (
                    <span className="px-2 py-1 bg-gray-500/30 rounded-lg text-xs border border-gray-500/30">
                      Coming Soon
                    </span>
                  )}
                </div>
              </div>
              <p className="text-gray-300 text-sm mb-2 leading-relaxed">{set.description}</p>
              <p className="text-gray-400 text-xs mb-6 font-mono">Dataset: {set.hfDataset}</p>
              
              <div className="flex gap-3">
                <button
                  onClick={() => runEvaluation(set.name)}
                  disabled={!set.available || !selectedGuard || (compareMode && !selectedGuard2) || evaluating}
                  className="flex-1 bg-white/10 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-sm px-4 py-2.5 rounded-xl font-medium text-sm transition-all hover:scale-105 active:scale-95 border border-white/20"
                >
                  {evaluating ? "Running..." : compareMode ? "Compare Guardrails" : "Run Evaluation"}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Results Display */}
      {results && (
        <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 animate-slide-up">
          <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-400" />
            Evaluation Results
          </h3>
          
          {/* Primary Metrics - Now Expandable */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {/* Correct Samples Card (True Negatives) */}
            <div 
              className="bg-black/20 backdrop-blur-sm border border-green-500/20 rounded-xl p-4 cursor-pointer hover:bg-black/30 transition-all"
              onClick={() => setExpandedCard(expandedCard === 'correct' ? null : 'correct')}
            >
              <div className="text-3xl font-bold text-green-400">{results.confusion_matrix?.tn || 0}</div>
              <div className="text-sm text-gray-400 mt-1">Correct</div>
              <div className="text-xs text-gray-500 mt-0.5">ALLOWED (Benign)</div>
              <div className="text-xs text-blue-400 mt-2">
                {expandedCard === 'correct' ? '▼ Hide' : '▶ Show samples'}
              </div>
            </div>
            
            {/* Incorrect Samples Card (False Positives) */}
            <div 
              className="bg-black/20 backdrop-blur-sm border border-red-500/20 rounded-xl p-4 cursor-pointer hover:bg-black/30 transition-all"
              onClick={() => setExpandedCard(expandedCard === 'incorrect' ? null : 'incorrect')}
            >
              <div className="text-3xl font-bold text-red-400">{results.confusion_matrix?.fp || 0}</div>
              <div className="text-sm text-gray-400 mt-1">Incorrect</div>
              <div className="text-xs text-gray-500 mt-0.5">BLOCKED (False Positive)</div>
              <div className="text-xs text-blue-400 mt-2">
                {expandedCard === 'incorrect' ? '▼ Hide' : '▶ Show samples'}
              </div>
            </div>
            
            <div className="bg-black/20 backdrop-blur-sm border border-blue-500/20 rounded-xl p-4">
              <div className="text-3xl font-bold text-blue-400">{results.f1_score || 0}</div>
              <div className="text-sm text-gray-400 mt-1">F1 Score</div>
              <div className="text-xs text-gray-500 mt-0.5">Precision × Recall</div>
            </div>
            <div className="bg-black/20 backdrop-blur-sm border border-purple-500/20 rounded-xl p-4">
              <div className="text-3xl font-bold text-purple-400">{results.avg_latency_ms}ms</div>
              <div className="text-sm text-gray-400 mt-1">Avg Latency</div>
              <div className="text-xs text-gray-500 mt-0.5">Per sample</div>
            </div>
          </div>

          {/* Expandable Sample Lists */}
          {expandedCard === 'correct' && results.correct_samples && results.correct_samples.length > 0 && (
            <div className="mb-6 bg-green-500/5 backdrop-blur-sm border border-green-500/20 rounded-xl p-4 animate-slide-up">
              <h4 className="text-sm font-semibold text-green-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Correct Classifications ({results.correct_samples.length})
              </h4>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {results.correct_samples.map((sample: any) => (
                  <div key={sample.id} className="bg-black/30 backdrop-blur-sm border border-green-500/20 rounded-lg p-3 flex justify-between items-start gap-4">
                    <div className="flex-1">
                      <div className="text-xs text-gray-500 mb-1">Sample #{sample.id}</div>
                      <div className="text-sm text-gray-300 font-mono">{sample.prompt}</div>
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span className="text-xs text-gray-500">{sample.latency_ms}ms</span>
                      <span className="px-2 py-1 rounded-lg text-xs font-bold bg-green-500/20 text-green-300 border border-green-500/30">
                        {sample.verdict}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {expandedCard === 'incorrect' && results.incorrect_samples && results.incorrect_samples.length > 0 && (
            <div className="mb-6 bg-red-500/5 backdrop-blur-sm border border-red-500/20 rounded-xl p-4 animate-slide-up">
              <h4 className="text-sm font-semibold text-red-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                <XCircle className="w-4 h-4" />
                Incorrect Classifications ({results.incorrect_samples.length})
              </h4>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {results.incorrect_samples.map((sample: any) => (
                  <div key={sample.id} className="bg-black/30 backdrop-blur-sm border border-red-500/20 rounded-lg p-3 flex justify-between items-start gap-4">
                    <div className="flex-1">
                      <div className="text-xs text-gray-500 mb-1">Sample #{sample.id}</div>
                      <div className="text-sm text-gray-300 font-mono">{sample.prompt}</div>
                    </div>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span className="text-xs text-gray-500">{sample.latency_ms}ms</span>
                      <span className="px-2 py-1 rounded-lg text-xs font-bold bg-red-500/20 text-red-300 border border-red-500/30">
                        {sample.verdict}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Secondary Metrics */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl p-3">
              <div className="text-xl font-bold text-white">{results.accuracy}%</div>
              <div className="text-xs text-gray-400 mt-1">Accuracy</div>
            </div>
            <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-xl p-3">
              <div className="text-xl font-bold text-white">{results.total_samples || 0}</div>
              <div className="text-xs text-gray-400 mt-1">Total Samples</div>
            </div>
          </div>

          {/* Advanced Metrics: FPR, Precision, Recall */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 backdrop-blur-sm border border-red-500/20 rounded-xl p-4">
              <div className="text-2xl font-bold text-red-400">{results.false_positive_rate}%</div>
              <div className="text-sm text-gray-300 mt-1 font-semibold">False Positive Rate</div>
              <div className="text-xs text-gray-500 mt-0.5">% Benign blocked (lower is better)</div>
            </div>
            <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-sm border border-blue-500/20 rounded-xl p-4">
              <div className="text-2xl font-bold text-blue-400">{results.precision}%</div>
              <div className="text-sm text-gray-300 mt-1 font-semibold">Precision</div>
              <div className="text-xs text-gray-500 mt-0.5">TP / (TP + FP)</div>
            </div>
            <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-sm border border-green-500/20 rounded-xl p-4">
              <div className="text-2xl font-bold text-green-400">{results.recall}%</div>
              <div className="text-sm text-gray-300 mt-1 font-semibold">Recall</div>
              <div className="text-xs text-gray-500 mt-0.5">TP / (TP + FN)</div>
            </div>
          </div>

          {/* Confusion Matrix */}
          {results.confusion_matrix && (
            <div className="bg-gradient-to-br from-purple-500/5 to-pink-500/5 backdrop-blur-sm border border-purple-500/20 rounded-xl p-5 mb-6">
              <h4 className="text-sm font-semibold text-purple-300 uppercase tracking-wider mb-4">Confusion Matrix</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-green-400">{results.confusion_matrix.tn}</div>
                  <div className="text-xs text-gray-400 mt-1 font-semibold">True Negatives (TN)</div>
                  <div className="text-xs text-gray-500 mt-0.5">Correctly allowed benign</div>
                </div>
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-red-400">{results.confusion_matrix.fp}</div>
                  <div className="text-xs text-gray-400 mt-1 font-semibold">False Positives (FP)</div>
                  <div className="text-xs text-gray-500 mt-0.5">Wrongly blocked benign</div>
                </div>
                <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-orange-400">{results.confusion_matrix.fn}</div>
                  <div className="text-xs text-gray-400 mt-1 font-semibold">False Negatives (FN)</div>
                  <div className="text-xs text-gray-500 mt-0.5">Wrongly allowed attacks</div>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-blue-400">{results.confusion_matrix.tp}</div>
                  <div className="text-xs text-gray-400 mt-1 font-semibold">True Positives (TP)</div>
                  <div className="text-xs text-gray-500 mt-0.5">Correctly blocked attacks</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Comparison Results Display */}
      {compareResults && (
        <div className="relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 animate-slide-up">
          <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
            <Shield className="w-6 h-6 text-purple-400" />
            Comparison Results
          </h3>
          
          {/* Guard Names Header */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 text-center">
              <div className="text-lg font-bold text-blue-400">{compareResults.guard1?.guard_name}</div>
              <div className="text-xs text-gray-400">Guardrail 1</div>
            </div>
            <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4 text-center">
              <div className="text-lg font-bold text-purple-400">{compareResults.guard2?.guard_name}</div>
              <div className="text-xs text-gray-400">Guardrail 2</div>
            </div>
          </div>

          {/* Side-by-Side Metrics */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Performance Metrics</h4>
            <div className="space-y-3">
              {[
                { label: "Accuracy", key: "accuracy", unit: "%" },
                { label: "F1 Score", key: "f1_score", unit: "" },
                { label: "False Positive Rate", key: "false_positive_rate", unit: "%" },
                { label: "Precision", key: "precision", unit: "%" },
                { label: "Recall", key: "recall", unit: "%" },
                { label: "Avg Latency", key: "avg_latency_ms", unit: "ms" }
              ].map((metric) => {
                const val1 = compareResults.guard1?.[metric.key] || 0;
                const val2 = compareResults.guard2?.[metric.key] || 0;
                const diff = compareResults.metric_diff?.[metric.key] || 0;
                const isLatency = metric.key === "avg_latency_ms";
                const isFPR = metric.key === "false_positive_rate";
                // For latency and FPR, lower is better; for others, higher is better
                const lowerIsBetter = isLatency || isFPR;
                const winner = diff === 0 ? null : (lowerIsBetter ? (diff < 0 ? 1 : 2) : (diff > 0 ? 1 : 2));
                
                return (
                  <div key={metric.key} className="grid grid-cols-3 gap-4 items-center">
                    <div className={`text-right p-3 rounded-lg ${winner === 1 ? "bg-green-500/10 border border-green-500/30" : "bg-black/20"}`}>
                      <span className="text-lg font-bold text-white">{val1}{metric.unit}</span>
                      {winner === 1 && <span className="ml-2 text-green-400 text-xs">✓ Better</span>}
                    </div>
                    <div className="text-center">
                      <div className="text-xs text-gray-400 mb-1">{metric.label}</div>
                      <div className={`text-sm font-mono ${diff === 0 ? "text-gray-500" : diff > 0 ? (lowerIsBetter ? "text-red-400" : "text-green-400") : (lowerIsBetter ? "text-green-400" : "text-red-400")}`}>
                        {diff > 0 ? "+" : ""}{diff}{metric.unit}
                      </div>
                    </div>
                    <div className={`text-left p-3 rounded-lg ${winner === 2 ? "bg-green-500/10 border border-green-500/30" : "bg-black/20"}`}>
                      <span className="text-lg font-bold text-white">{val2}{metric.unit}</span>
                      {winner === 2 && <span className="ml-2 text-green-400 text-xs">✓ Better</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Differences Summary */}
          <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border border-orange-500/20 rounded-xl p-5 mb-6">
            <h4 className="text-sm font-semibold text-orange-300 uppercase tracking-wider mb-3">
              Verdict Differences ({compareResults.difference_count} samples)
            </h4>
            <p className="text-sm text-gray-400 mb-4">
              Out of {compareResults.total_samples} samples, the two guardrails disagreed on {compareResults.difference_count} samples.
            </p>
            
            {compareResults.differences && compareResults.differences.length > 0 && (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {compareResults.differences.slice(0, 20).map((diff: any) => (
                  <div key={diff.sample_id} className="bg-black/30 backdrop-blur-sm border border-orange-500/20 rounded-lg p-3">
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex-1">
                        <div className="text-xs text-gray-500 mb-1">Sample #{diff.sample_id}</div>
                        <div className="text-sm text-gray-300 font-mono truncate">{diff.prompt}</div>
                      </div>
                      <div className="flex gap-2 flex-shrink-0">
                        <span className={`px-2 py-1 rounded-lg text-xs font-bold ${
                          diff.guard1_verdict === "ALLOWED" 
                            ? "bg-green-500/20 text-green-300 border border-green-500/30" 
                            : "bg-red-500/20 text-red-300 border border-red-500/30"
                        }`}>
                          G1: {diff.guard1_verdict}
                        </span>
                        <span className={`px-2 py-1 rounded-lg text-xs font-bold ${
                          diff.guard2_verdict === "ALLOWED" 
                            ? "bg-green-500/20 text-green-300 border border-green-500/30" 
                            : "bg-red-500/20 text-red-300 border border-red-500/30"
                        }`}>
                          G2: {diff.guard2_verdict}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
                {compareResults.differences.length > 20 && (
                  <div className="text-center text-sm text-gray-500 py-2">
                    ... and {compareResults.differences.length - 20} more differences
                  </div>
                )}
              </div>
            )}
            
            {(!compareResults.differences || compareResults.differences.length === 0) && (
              <div className="text-center text-green-400 py-4">
                <CheckCircle className="w-8 h-8 mx-auto mb-2" />
                Both guardrails gave identical verdicts on all samples!
              </div>
            )}
          </div>

          {/* Side-by-Side Confusion Matrices */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl p-4">
              <h5 className="text-sm font-semibold text-blue-300 mb-3">Guard 1 Confusion Matrix</h5>
              <div className="grid grid-cols-2 gap-2 text-center text-sm">
                <div className="bg-green-500/10 rounded-lg p-2">
                  <div className="font-bold text-green-400">{compareResults.guard1?.confusion_matrix?.tn || 0}</div>
                  <div className="text-xs text-gray-500">TN</div>
                </div>
                <div className="bg-red-500/10 rounded-lg p-2">
                  <div className="font-bold text-red-400">{compareResults.guard1?.confusion_matrix?.fp || 0}</div>
                  <div className="text-xs text-gray-500">FP</div>
                </div>
                <div className="bg-orange-500/10 rounded-lg p-2">
                  <div className="font-bold text-orange-400">{compareResults.guard1?.confusion_matrix?.fn || 0}</div>
                  <div className="text-xs text-gray-500">FN</div>
                </div>
                <div className="bg-blue-500/10 rounded-lg p-2">
                  <div className="font-bold text-blue-400">{compareResults.guard1?.confusion_matrix?.tp || 0}</div>
                  <div className="text-xs text-gray-500">TP</div>
                </div>
              </div>
            </div>
            <div className="bg-purple-500/5 border border-purple-500/20 rounded-xl p-4">
              <h5 className="text-sm font-semibold text-purple-300 mb-3">Guard 2 Confusion Matrix</h5>
              <div className="grid grid-cols-2 gap-2 text-center text-sm">
                <div className="bg-green-500/10 rounded-lg p-2">
                  <div className="font-bold text-green-400">{compareResults.guard2?.confusion_matrix?.tn || 0}</div>
                  <div className="text-xs text-gray-500">TN</div>
                </div>
                <div className="bg-red-500/10 rounded-lg p-2">
                  <div className="font-bold text-red-400">{compareResults.guard2?.confusion_matrix?.fp || 0}</div>
                  <div className="text-xs text-gray-500">FP</div>
                </div>
                <div className="bg-orange-500/10 rounded-lg p-2">
                  <div className="font-bold text-orange-400">{compareResults.guard2?.confusion_matrix?.fn || 0}</div>
                  <div className="text-xs text-gray-500">FN</div>
                </div>
                <div className="bg-blue-500/10 rounded-lg p-2">
                  <div className="font-bold text-blue-400">{compareResults.guard2?.confusion_matrix?.tp || 0}</div>
                  <div className="text-xs text-gray-500">TP</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="relative bg-blue-500/5 backdrop-blur-sm border border-blue-500/20 rounded-2xl p-6 overflow-hidden group hover:border-blue-500/30 transition-all">
        <div className="absolute top-0 right-0 w-48 h-48 bg-blue-500/10 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
        <div className="relative flex items-start gap-4">
          <div className="p-2 bg-blue-500/20 rounded-xl flex-shrink-0">
            <AlertTriangle className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h4 className="font-semibold text-blue-300 mb-2">About NotInject Dataset</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-2">
              The <span className="font-mono text-blue-300">NotInject</span> dataset contains benign prompts (non-injection scenarios). 
              The evaluation measures how accurately your guardrail identifies safe content:
            </p>
            <ul className="text-xs text-gray-500 space-y-1 ml-4 list-disc">
              <li><span className="text-green-400">Correct</span>: Benign prompts that were ALLOWED (True Positives)</li>
              <li><span className="text-red-400">Incorrect</span>: Benign prompts that were BLOCKED (False Positives)</li>
              <li><span className="text-blue-400">F1 Score</span>: Harmonic mean of precision and recall (0-1)</li>
              <li><span className="text-purple-400">Latency</span>: Average time per classification in milliseconds</li>
            </ul>
            <p className="text-xs text-gray-500 mt-3">
              Note: Evaluation runs on first 100 samples by default. Adjust max_samples in the API for full evaluation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// --- COMPONENT: MAIN PAGE ---
export default function Dashboard() {
  const [guards, setGuards] = useState<Guard[]>([]);
  const [mode, setMode] = useState<"build" | "evaluate">("build");
  const [loading, setLoading] = useState(true);

  // Lifted state for BuildMode (cached when switching modes)
  const [buildModeState, setBuildModeState] = useState<BuildModeState>({
    activeGuard: null,
    showBuilder: false,
    editingGuard: null,
    // InteractiveTester state
    testerPrompt: "",
    testerResult: null,
    testerHfToken: "",
    testerShowOptions: false
  });

  // Lifted state for EvaluationMode (cached when switching modes)
  const [evalModeState, setEvalModeState] = useState<EvaluationModeState>({
    selectedGuard: null,
    selectedGuard2: null,
    results: null,
    compareResults: null,
    hfToken: "",
    showTokenInput: false,
    compareMode: false,
    expandedCard: null
  });

  useEffect(() => {
    fetchGuards();
  }, []);

  const fetchGuards = async () => {
    try {
      const res = await api.get("/guards");
      setGuards(res.data);
    } catch (error) {
      console.error("Failed to fetch guardrails:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      {/* Mode Toggle */}
      <div className="flex justify-center items-center gap-3 pt-8">
        <button
          onClick={() => setMode("build")}
          className={`px-6 py-2.5 rounded-xl font-semibold transition-all ${
            mode === "build"
              ? "bg-blue-500 text-white shadow-lg shadow-blue-500/50"
              : "bg-white/5 text-gray-400 hover:bg-white/10"
          }`}
        >
          🔨 Build Mode
        </button>
        <button
          onClick={() => setMode("evaluate")}
          className={`px-6 py-2.5 rounded-xl font-semibold transition-all ${
            mode === "evaluate"
              ? "bg-purple-500 text-white shadow-lg shadow-purple-500/50"
              : "bg-white/5 text-gray-400 hover:bg-white/10"
          }`}
        >
          📊 Evaluate Mode
        </button>
      </div>

      <div className="container mx-auto px-6 py-8">
        {loading ? (
          <div className="text-center text-gray-400 py-12">Loading...</div>
        ) : mode === "build" ? (
          <BuildMode 
            guards={guards} 
            refreshGuards={fetchGuards} 
            state={buildModeState}
            setState={setBuildModeState}
          />
        ) : (
          <EvaluationMode 
            guards={guards} 
            state={evalModeState}
            setState={setEvalModeState}
          />
        )}
      </div>
    </div>
  );
}