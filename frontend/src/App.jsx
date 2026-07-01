import React, { useState, useEffect, useRef } from 'react'
import { 
  Play, StopCircle, RefreshCw, Trash2, Download, 
  Settings, Layers, Edit, Activity, FileText, CheckCircle2, AlertTriangle, FileCode,
  UploadCloud, Terminal, Server, Cpu
} from 'lucide-react'

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard') // 'dashboard', 'projects', 'editor', 'optimizer', 'logs'
  
  // State from Backend API
  const [systemStatus, setSystemStatus] = useState({
    cuda_available: false,
    device_name: "None",
    current_job: null,
    queue: []
  })
  
  // System Log and Resource Status States
  const [logType, setLogType] = useState('worker')
  const [logContent, setLogContent] = useState('')
  const [tailLines, setTailLines] = useState(500)
  const [autoRefreshLogs, setAutoRefreshLogs] = useState(false)
  const [detailedStatus, setDetailedStatus] = useState(null)
  const [isLogsLoading, setIsLogsLoading] = useState(false)
  
  // Realtime chart data
  const [realtimeData, setRealtimeData] = useState([])
  
  // Project list and selection
  const [projects, setProjects] = useState([])
  const [selectedProject, setSelectedProject] = useState('')
  const [projectDetails, setProjectDetails] = useState(null)
  
  // Forms & parameters
  const [nptParams, setNptParams] = useState({
    selectedModel: 'CHGNet',
    simMode: 'Realistic (ISIF=3)',
    projectPrefix: '',
    magmomSpecies: ['Co'],
    tempStart: 10,
    tempEnd: 1000,
    tempStep: 5,
    eqSteps: 2000,
    nGpuJobs: 1,
    enableCooling: false
  })

  const [newSpecie, setNewSpecie] = useState('')
  
  const [optParams, setOptParams] = useState({
    selectedModel: 'CHGNet',
    projectPrefix: '',
    fmax: 0.01
  })
  
  const [optRealtimeData, setOptRealtimeData] = useState([])
  
  const [cifEditorParams, setCifEditorParams] = useState({
    operation: 'Substitution', // 'Substitution' or 'Vacancy'
    replaceFrom: 'Bi',
    replaceTo: 'Mg',
    percentage: 10,
    mode: 'random' // 'random' or 'sequential'
  })
  
  // Uploader Files State
  const [nptFiles, setNptFiles] = useState([])
  const [optFiles, setOptFiles] = useState([])
  const [editorFile, setEditorFile] = useState(null)
  const [editedCifFile, setEditedCifFile] = useState(null)
  const [editorLogs, setEditorLogs] = useState([])
  const [nptDragActive, setNptDragActive] = useState(false)
  const [optDragActive, setOptDragActive] = useState(false)
  const [editorDragActive, setEditorDragActive] = useState(false)

  // UI state
  const [notification, setNotification] = useState(null)
  const [confirmRestart, setConfirmRestart] = useState(false)
  
  // Refs for file inputs
  const nptFileInputRef = useRef(null)
  const optFileInputRef = useRef(null)
  const editorFileInputRef = useRef(null)

  // Fetch queue status and system info
  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status')
      if (res.ok) {
        const data = await res.json()
        setSystemStatus(data)
      }
    } catch (err) {
      console.error("Failed to fetch status:", err)
    }
  }

  // Fetch realtime data for current job
  const fetchRealtime = async () => {
    if (!systemStatus.current_job) {
      setRealtimeData([])
      return
    }
    try {
      const res = await fetch('/api/jobs/realtime')
      if (res.ok) {
        const data = await res.json()
        setRealtimeData(data)
      }
    } catch (err) {
      console.error("Failed to fetch realtime data:", err)
    }
  }

  // Fetch realtime data for structure optimization
  const fetchOptRealtime = async () => {
    if (!systemStatus.current_job) {
      setOptRealtimeData([])
      return
    }
    try {
      const res = await fetch('/api/jobs/opt_realtime')
      if (res.ok) {
        const data = await res.json()
        setOptRealtimeData(data)
      }
    } catch (err) {
      console.error("Failed to fetch opt realtime data:", err)
    }
  }

  // Fetch list of saved projects
  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects')
      if (res.ok) {
        const data = await res.json()
        setProjects(data.projects || [])
      }
    } catch (err) {
      console.error("Failed to fetch projects list:", err)
    }
  }

  // Fetch selected project details
  const fetchProjectDetails = async (projectName) => {
    if (!projectName) return
    try {
      const res = await fetch(`/api/projects/${projectName}`)
      if (res.ok) {
        const data = await res.json()
        setProjectDetails(data)
      }
    } catch (err) {
      console.error("Failed to fetch project details:", err)
    }
  }

  const fetchSystemLogs = async (type = logType, tail = tailLines) => {
    try {
      setIsLogsLoading(true)
      const res = await fetch(`/api/system/logs/${type}?tail=${tail}`)
      if (res.ok) {
        const data = await res.json()
        setLogContent(data.content)
      } else {
        setLogContent(`Error fetching logs: ${res.statusText}`)
      }
    } catch (err) {
      setLogContent(`Network error fetching logs: ${err.message}`)
    } finally {
      setIsLogsLoading(false)
    }
  }

  const fetchDetailedStatus = async () => {
    try {
      const res = await fetch('/api/system/status')
      if (res.ok) {
        const data = await res.json()
        setDetailedStatus(data)
      }
    } catch (err) {
      console.error("Error fetching detailed system status:", err)
    }
  }

  const clearSystemLog = async (type = logType) => {
    if (!window.confirm(`Are you sure you want to clear/truncate the ${type} log?`)) return
    try {
      const res = await fetch(`/api/system/logs/${type}/clear`, { method: 'POST' })
      if (res.ok) {
        alert(`Log ${type} cleared successfully.`)
        fetchSystemLogs(type, tailLines)
      } else {
        const data = await res.json()
        alert(`Failed to clear log: ${data.detail || res.statusText}`)
      }
    } catch (err) {
      alert(`Error clearing log: ${err.message}`)
    }
  }

  // Polling for system logs and status when log tab is active or auto refresh is enabled
  useEffect(() => {
    if (activeTab === 'logs') {
      fetchSystemLogs(logType, tailLines)
      fetchDetailedStatus()
    }
  }, [activeTab, logType, tailLines])

  useEffect(() => {
    let intervalId = null
    if (activeTab === 'logs' && autoRefreshLogs) {
      intervalId = setInterval(() => {
        fetchSystemLogs(logType, tailLines)
        fetchDetailedStatus()
      }, 5000)
    }
    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [activeTab, logType, tailLines, autoRefreshLogs])

  // Periodic polling
  useEffect(() => {
    fetchStatus()
    fetchProjects()
    
    const interval = setInterval(() => {
      fetchStatus()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (systemStatus.current_job) {
      const isOpt = systemStatus.current_job.job_type === 'optimize_only'
      if (isOpt) {
        fetchOptRealtime()
      } else {
        fetchRealtime()
      }
      const chartInterval = setInterval(() => {
        if (isOpt) {
          fetchOptRealtime()
        } else {
          fetchRealtime()
        }
      }, 3000)
      return () => clearInterval(chartInterval)
    } else {
      setRealtimeData([])
      setOptRealtimeData([])
    }
  }, [systemStatus.current_job])

  useEffect(() => {
    if (selectedProject) {
      fetchProjectDetails(selectedProject)
    } else {
      setProjectDetails(null)
    }
  }, [selectedProject])

  const handleAddSpecie = (e) => {
    e.preventDefault()
    const val = newSpecie.trim()
    if (val && !nptParams.magmomSpecies.includes(val)) {
      setNptParams(prev => ({
        ...prev,
        magmomSpecies: [...prev.magmomSpecies, val]
      }))
    }
    setNewSpecie('')
  }

  const handleSpecieKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      const val = newSpecie.trim()
      if (val && !nptParams.magmomSpecies.includes(val)) {
        setNptParams(prev => ({
          ...prev,
          magmomSpecies: [...prev.magmomSpecies, val]
        }))
      }
      setNewSpecie('')
    }
  }

  const handleRemoveSpecie = (specieToRemove) => {
    setNptParams(prev => ({
      ...prev,
      magmomSpecies: prev.magmomSpecies.filter(s => s !== specieToRemove)
    }))
  }

  const handleNptNumberChange = (field, val) => {
    if (val === "") {
      setNptParams(prev => ({ ...prev, [field]: "" }))
      return
    }
    const parsed = parseInt(val, 10)
    setNptParams(prev => ({ ...prev, [field]: isNaN(parsed) ? "" : parsed }))
  }

  const handleEditorPercentageChange = (val) => {
    if (val === "") {
      setCifEditorParams(prev => ({ ...prev, percentage: "" }))
      return
    }
    const parsed = parseFloat(val)
    setCifEditorParams(prev => ({ ...prev, percentage: isNaN(parsed) ? "" : parsed }))
  }

  const isNptFormInvalid = 
    nptParams.tempStart === "" ||
    nptParams.tempEnd === "" ||
    nptParams.tempStep === "" ||
    nptParams.eqSteps === "" ||
    nptParams.nGpuJobs === "" ||
    nptFiles.length === 0

  const getChartDataWithTemp = () => {
    if (!realtimeData || realtimeData.length === 0) return []
    const batches = {}
    realtimeData.forEach((d) => {
      const t = d.set_temps
      if (!batches[t]) {
        batches[t] = []
      }
      batches[t].push(d)
    })
    const sortedTemps = Object.keys(batches).map(Number).sort((a, b) => a - b)
    const jobParams = systemStatus.current_job?.params
    const startTemp = jobParams?.temp_range?.[0] || nptParams.tempStart || 10
    const stepTemp = jobParams?.temp_range?.[2] || nptParams.tempStep || 5

    return realtimeData.map((d) => {
      const t_curr = d.set_temps
      const batch = batches[t_curr]
      const batchIndex = batch.indexOf(d)
      const N = batch.length || 1
      const tempIdx = sortedTemps.indexOf(t_curr)
      let t_prev = startTemp - stepTemp
      if (tempIdx > 0) {
        t_prev = sortedTemps[tempIdx - 1]
      } else {
        t_prev = t_curr - stepTemp
      }
      const interpTemp = t_prev + (t_curr - t_prev) * ((batchIndex + 1) / N)
      return {
        ...d,
        calculatedTemp: parseFloat(interpTemp.toFixed(1))
      }
    })
  }

  const handleDrag = (e, setDragActive) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e, setDragActive, setFiles, multiple = true) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const filesArr = Array.from(e.dataTransfer.files)
      const cifFiles = filesArr.filter(file => file.name.endsWith('.cif'))
      if (cifFiles.length > 0) {
        if (multiple) {
          setFiles(cifFiles)
        } else {
          setFiles(cifFiles[0])
        }
      } else {
        showToast("Please drop valid .cif files only.", "warning")
      }
    }
  }

  // Show temporary toast notification
  const showToast = (message, type = 'info') => {
    setNotification({ message, type })
    setTimeout(() => setNotification(null), 5000)
  }

  // Handle NPT simulation queue submit
  const handleAddNptJobs = async (e) => {
    e.preventDefault()
    if (nptFiles.length === 0) {
      showToast("Please select at least one CIF file.", "warning")
      return
    }
    
    const formData = new FormData()
    nptFiles.forEach(file => {
      formData.append("files", file)
    })
    formData.append("model", nptParams.selectedModel)
    formData.append("sim_mode", nptParams.simMode)
    formData.append("project_prefix", nptParams.projectPrefix || new Date().toISOString().slice(0,10).replace(/-/g,""))
    formData.append("magmom_specie", nptParams.magmomSpecies.join(","))
    formData.append("temp_start", nptParams.tempStart)
    formData.append("temp_end", nptParams.tempEnd)
    formData.append("temp_step", nptParams.tempStep)
    formData.append("eq_steps", nptParams.eqSteps)
    formData.append("n_gpu_jobs", nptParams.nGpuJobs)
    formData.append("enable_cooling", nptParams.enableCooling)

    try {
      const res = await fetch('/api/jobs/npt', {
        method: 'POST',
        body: formData
      })
      const result = await res.json()
      if (res.ok) {
        showToast(result.message, "success")
        setNptFiles([])
        if (nptFileInputRef.current) nptFileInputRef.current.value = ""
        fetchStatus()
      } else {
        showToast(`Error: ${result.detail || "Failed to add NPT jobs."}`, "danger")
      }
    } catch (err) {
      showToast("Network error submitting NPT jobs.", "danger")
    }
  }

  // Handle structure optimization queue submit
  const handleAddOptJob = async (e) => {
    e.preventDefault()
    if (optFiles.length === 0) {
      showToast("Please select a CIF file for optimization.", "warning")
      return
    }

    const formData = new FormData()
    optFiles.forEach(file => {
      formData.append("files", file)
    })
    formData.append("model", optParams.selectedModel)
    formData.append("project_prefix", optParams.projectPrefix || new Date().toISOString().slice(0,10).replace(/-/g,""))
    formData.append("fmax", optParams.fmax || 0.01)

    try {
      const res = await fetch('/api/jobs/optimize', {
        method: 'POST',
        body: formData
      })
      const result = await res.json()
      if (res.ok) {
        showToast(result.message, "success")
        setOptFiles([])
        if (optFileInputRef.current) optFileInputRef.current.value = ""
        fetchStatus()
      } else {
        showToast(`Error: ${result.detail || "Failed to add optimization job."}`, "danger")
      }
    } catch (err) {
      showToast("Network error submitting optimization job.", "danger")
    }
  }

  // Cancel currently running job
  const handleCancelJob = async () => {
    if (!confirm("Are you sure you want to stop the current running job?")) return
    try {
      const res = await fetch('/api/jobs/cancel', { method: 'POST' })
      if (res.ok) {
        showToast("Cancellation request issued. Job stopping shortly.", "warning")
        fetchStatus()
      }
    } catch (err) {
      showToast("Failed to request job cancellation.", "danger")
    }
  }

  // Delete jobs from waiting queue
  const handleDeleteQueueJob = async (jobIndex) => {
    if (!confirm(`Remove job #${jobIndex} from the queue?`)) return
    try {
      const res = await fetch('/api/jobs/queue', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: jobIndex })
      })
      if (res.ok) {
        showToast("Job removed from queue.", "success")
        fetchStatus()
      }
    } catch (err) {
      showToast("Failed to remove job from queue.", "danger")
    }
  }

  // Trigger app reload/maintenance restart command
  const handleForceRestart = async () => {
    setConfirmRestart(false)
    try {
      const res = await fetch('/api/maintenance/restart', { method: 'POST' })
      if (res.ok) {
        showToast("Maintenance restart command issued. Reconnecting in 15s...", "warning")
        setTimeout(() => window.location.reload(), 15000)
      } else {
        showToast("Failed to run restart script.", "danger")
      }
    } catch (err) {
      showToast("Network error executing restart.", "danger")
    }
  }

  // Delete saved project
  const handleDeleteProject = async (projectName) => {
    if (!confirm(`Are you sure you want to permanently delete project '${projectName}'?`)) return
    try {
      const res = await fetch(`/api/projects/${projectName}`, { method: 'DELETE' })
      if (res.ok) {
        showToast(`Project '${projectName}' deleted.`, "success")
        setSelectedProject('')
        fetchProjects()
      }
    } catch (err) {
      showToast("Failed to delete project.", "danger")
    }
  }

  // Create or update analysis ZIP
  const handleCreateZip = async (projectName) => {
    showToast("Zipping analysis files in progress...", "info")
    try {
      const res = await fetch(`/api/projects/${projectName}/zip`, { method: 'POST' })
      if (res.ok) {
        showToast("Analysis ZIP packaged successfully!", "success")
        fetchProjectDetails(projectName)
      } else {
        showToast("Failed to create ZIP package.", "danger")
      }
    } catch (err) {
      showToast("Error creating ZIP package.", "danger")
    }
  }

  // Process CIF element edit / vacancy creation
  const handleCifEditSubmit = async (e) => {
    e.preventDefault()
    if (!editorFile) {
      showToast("Please upload a CIF file for editing first.", "warning")
      return
    }

    const formData = new FormData()
    formData.append("file", editorFile)
    formData.append("operation", cifEditorParams.operation)
    formData.append("replace_from", cifEditorParams.replaceFrom)
    formData.append("replace_to", cifEditorParams.operation === 'Vacancy' ? '' : cifEditorParams.replaceTo)
    formData.append("percentage", cifEditorParams.percentage)
    formData.append("mode", cifEditorParams.mode)

    try {
      const res = await fetch('/api/cif/edit', {
        method: 'POST',
        body: formData
      })
      const result = await res.json()
      if (res.ok) {
        setEditedCifFile({
          name: result.new_filename,
          content: result.content
        })
        setEditorLogs(result.logs || [])
        showToast("CIF processing finished!", "success")
      } else {
        showToast(result.detail || "Error processing CIF.", "danger")
      }
    } catch (err) {
      showToast("Network error editing CIF.", "danger")
    }
  }

  // Trigger file download helper
  const downloadFile = (filename, content, mime = "text/plain") => {
    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app-container">
      
      {/* Toast Notifications */}
      {notification && (
        <div style={{
          position: 'fixed', top: '24px', right: '24px', zIndex: 1000,
          minWidth: '320px', transition: 'all 0.2s ease'
        }} className={`alert alert-${notification.type}`}>
          {notification.message}
        </div>
      )}

      {/* Sidebar - NPT Parameter Settings */}
      <aside className="sidebar">
        <div className="logo-section">
          <Layers size={24} className="pulse" style={{ color: 'var(--color-primary)' }} />
          <h1>Universal MD Simulator</h1>
        </div>
        
        {/* GPU Status Info */}
        <div className={`gpu-badge ${systemStatus.cuda_available ? 'active' : 'inactive'}`}>
          <Activity size={16} />
          <div>
            {systemStatus.cuda_available ? (
              <span>GPU Active: {systemStatus.device_name}</span>
            ) : (
              <span>CPU-only Mode (CUDA inactive)</span>
            )}
          </div>
        </div>

        {activeTab === 'dashboard' && (
          <form className="sidebar-form" onSubmit={handleAddNptJobs}>
            <h3>NPT Simulation Settings</h3>
            
            <div className="form-group">
              <label>ML Force Field</label>
              <select className="form-select" value={nptParams.selectedModel}
                onChange={e => setNptParams({...nptParams, selectedModel: e.target.value})}>
                <option value="CHGNet">CHGNet</option>
                <option value="matgl_chgnet_r2scan">CHGNet r2SCAN (MatGL)</option>
                <option value="matris_10m_oam">MatRIS (matris_10m_oam)</option>
                <option value="matris_10m_mp">MatRIS (matris_10m_mp)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Simulation Mode</label>
              <select className="form-select" value={nptParams.simMode}
                onChange={e => setNptParams({...nptParams, simMode: e.target.value})}>
                <option value="Realistic (ISIF=3)">Realistic (ISIF=3)</option>
                <option value="Legacy (Orthorombic)">Legacy (Orthorombic)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Project Name Prefix</label>
              <input type="text" className="form-input" placeholder="e.g. 20260625" 
                value={nptParams.projectPrefix}
                onChange={e => setNptParams({...nptParams, projectPrefix: e.target.value})} />
            </div>

            <div className="form-group">
              <label>Magmom species tracking</label>
              <div className="tags-list">
                {nptParams.magmomSpecies.map((specie, index) => (
                  <span key={index} className="tag-chip">
                    {specie}
                    <button type="button" onClick={() => handleRemoveSpecie(specie)}>×</button>
                  </span>
                ))}
              </div>
              <div className="tag-input-row">
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="e.g. Co" 
                  value={newSpecie}
                  onChange={e => setNewSpecie(e.target.value)}
                  onKeyDown={handleSpecieKeyDown}
                />
                <button type="button" className="btn btn-secondary" onClick={handleAddSpecie}>+</button>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Start Temp (K)</label>
                <input 
                  type="number" 
                  className={`form-input ${nptParams.tempStart === "" ? "invalid" : ""}`} 
                  value={nptParams.tempStart}
                  onChange={e => handleNptNumberChange('tempStart', e.target.value)} 
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>End Temp (K)</label>
                <input 
                  type="number" 
                  className={`form-input ${nptParams.tempEnd === "" ? "invalid" : ""}`} 
                  value={nptParams.tempEnd}
                  onChange={e => handleNptNumberChange('tempEnd', e.target.value)} 
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Temp Step (K)</label>
                <input 
                  type="number" 
                  className={`form-input ${nptParams.tempStep === "" ? "invalid" : ""}`} 
                  value={nptParams.tempStep}
                  onChange={e => handleNptNumberChange('tempStep', e.target.value)} 
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>Steps per Temp</label>
                <input 
                  type="number" 
                  className={`form-input ${nptParams.eqSteps === "" ? "invalid" : ""}`} 
                  value={nptParams.eqSteps}
                  onChange={e => handleNptNumberChange('eqSteps', e.target.value)} 
                />
              </div>
            </div>

            <div className="form-group">
              <label>Parallel GPU Jobs</label>
              <input 
                type="number" 
                className={`form-input ${nptParams.nGpuJobs === "" ? "invalid" : ""}`} 
                min="1" 
                max="8" 
                value={nptParams.nGpuJobs}
                onChange={e => handleNptNumberChange('nGpuJobs', e.target.value)} 
              />
            </div>

            <label className="form-checkbox">
              <input type="checkbox" checked={nptParams.enableCooling}
                onChange={e => setNptParams({...nptParams, enableCooling: e.target.checked})} />
              Enable Cooling after Heating
            </label>

             <div className="form-group">
               <label>Select CIF Files</label>
               <div 
                 className={`compact-uploader ${nptDragActive ? 'dragover' : ''} ${nptFiles.length === 0 ? 'invalid' : ''}`}
                 onDragEnter={e => handleDrag(e, setNptDragActive)}
                 onDragOver={e => handleDrag(e, setNptDragActive)}
                 onDragLeave={e => handleDrag(e, setNptDragActive)}
                 onDrop={e => handleDrop(e, setNptDragActive, setNptFiles, true)}
                 onClick={() => nptFileInputRef.current.click()}
               >
                 <UploadCloud size={20} className={nptFiles.length > 0 ? "active" : ""} />
                 <span>Drag & drop CIFs here</span>
                 <span className="subtext">or click to browse</span>
                 <input 
                   type="file" 
                   ref={nptFileInputRef} 
                   style={{ display: 'none' }} 
                   accept=".cif" 
                   multiple
                   onChange={e => setNptFiles(Array.from(e.target.files))} 
                 />
               </div>
               
               {nptFiles.length > 0 && (
                 <div className="uploaded-files-list-sidebar">
                   {nptFiles.map((file, idx) => (
                     <div key={idx} className="uploaded-file-item-sidebar">
                       <span>{file.name}</span>
                       <button type="button" onClick={(e) => { e.stopPropagation(); setNptFiles(prev => prev.filter((_, i) => i !== idx)); }}>×</button>
                     </div>
                   ))}
                 </div>
               )}
             </div>

            <button type="submit" className="btn btn-primary" style={{ marginTop: '12px' }} disabled={isNptFormInvalid}>
              <Play size={16} /> Add NPT Jobs to Queue
            </button>
          </form>
        )}

        {activeTab === 'optimizer' && (
          <form className="sidebar-form" onSubmit={handleAddOptJob}>
            <h3>Structure Optimizer Settings</h3>
            
            <div className="form-group">
              <label>ML Force Field</label>
              <select className="form-select" value={optParams.selectedModel}
                onChange={e => setOptParams({...optParams, selectedModel: e.target.value})}>
                <option value="CHGNet">CHGNet</option>
                <option value="matgl_chgnet_r2scan">CHGNet r2SCAN (MatGL)</option>
                <option value="matris_10m_oam">MatRIS (matris_10m_oam)</option>
                <option value="matris_10m_mp">MatRIS (matris_10m_mp)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Target fmax (eV/Å)</label>
              <input type="number" className="form-input" step="0.0001" min="0.0001" max="1.0"
                value={optParams.fmax}
                onChange={e => setOptParams({...optParams, fmax: parseFloat(e.target.value) || 0.01})} />
              <small style={{ color: 'var(--text-muted)', display: 'block', marginTop: '4px' }}>
                Convergence threshold. Default is 0.01.
              </small>
            </div>

            <div className="form-group">
              <label>Project Name Prefix</label>
              <input type="text" className="form-input" placeholder="e.g. 20260625" 
                value={optParams.projectPrefix}
                onChange={e => setOptParams({...optParams, projectPrefix: e.target.value})} />
            </div>

             <div className="form-group">
               <label>Select CIF File</label>
               <div 
                 className={`compact-uploader ${optDragActive ? 'dragover' : ''} ${optFiles.length === 0 ? 'invalid' : ''}`}
                 onDragEnter={e => handleDrag(e, setOptDragActive)}
                 onDragOver={e => handleDrag(e, setOptDragActive)}
                 onDragLeave={e => handleDrag(e, setOptDragActive)}
                 onDrop={e => handleDrop(e, setOptDragActive, setOptFiles, true)}
                 onClick={() => optFileInputRef.current.click()}
               >
                 <UploadCloud size={20} className={optFiles.length > 0 ? "active" : ""} />
                 <span>Drag & drop CIF here</span>
                 <span className="subtext">or click to browse</span>
                 <input 
                   type="file" 
                   ref={optFileInputRef} 
                   style={{ display: 'none' }} 
                   accept=".cif" 
                   multiple
                   onChange={e => setOptFiles(Array.from(e.target.files))} 
                 />
               </div>
               
               {optFiles.length > 0 && (
                 <div className="uploaded-files-list-sidebar">
                   {optFiles.map((file, idx) => (
                     <div key={idx} className="uploaded-file-item-sidebar">
                       <span>{file.name}</span>
                       <button type="button" onClick={(e) => { e.stopPropagation(); setOptFiles(prev => prev.filter((_, i) => i !== idx)); }}>×</button>
                     </div>
                   ))}
                 </div>
               )}
             </div>

            <button type="submit" className="btn btn-primary" style={{ marginTop: '12px' }} disabled={optFiles.length === 0}>
              <Settings size={16} /> Add OPT Job to Queue
            </button>
          </form>
        )}

        {/* Maintenance / Force Restart Section */}
        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
          {confirmRestart ? (
            <div className="dashboard-card" style={{ gap: '10px', padding: '16px' }}>
              <span style={{ fontSize: '12.5px', color: 'var(--color-warning)' }}>
                <strong>Confirm Maintenance Restart?</strong> This stops all active/queued tasks.
              </span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button className="btn btn-danger" style={{ padding: '8px 12px', fontSize: '12px' }} onClick={handleForceRestart}>Confirm</button>
                <button className="btn btn-secondary" style={{ padding: '8px 12px', fontSize: '12px' }} onClick={() => setConfirmRestart(false)}>Cancel</button>
              </div>
            </div>
          ) : (
            <button className="btn btn-secondary" onClick={() => setConfirmRestart(true)}>
              <RefreshCw size={14} /> Force Restart All Processes
            </button>
          )}
        </div>
      </aside>

      {/* Main Panel Content Area */}
      <main className="main-content">
        
        {/* Navigation Tabs */}
        <nav className="tabs-navigation">
          <button className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <Activity size={18} /> Simulation Dashboard
          </button>
          <button className={`tab-btn ${activeTab === 'optimizer' ? 'active' : ''}`} onClick={() => setActiveTab('optimizer')}>
            <Settings size={18} /> Structure Optimizer
          </button>
          <button className={`tab-btn ${activeTab === 'projects' ? 'active' : ''}`} onClick={() => { setActiveTab('projects'); fetchProjects(); }}>
            <FileText size={18} /> Project Browser
          </button>
          <button className={`tab-btn ${activeTab === 'editor' ? 'active' : ''}`} onClick={() => setActiveTab('editor')}>
            <Edit size={18} /> CIF Structure Editor
          </button>
          <button className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
            <Terminal size={18} /> System Logs & Status
          </button>
        </nav>

        {/* Tab Panels */}
        <div className="tab-panel">
          
          {/* 1. Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div>
              <div className="dashboard-grid">
                
                {/* Now Running Job */}
                <div className="dashboard-card">
                  <h3><Play size={18} style={{ color: 'var(--color-success)' }} /> Now Running</h3>
                  {systemStatus.current_job ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div className="alert alert-info" style={{ margin: 0 }}>
                        <strong>{systemStatus.current_job.project_name}</strong><br/>
                        <span style={{ fontSize: '12px', opacity: 0.8 }}>
                          Type: {systemStatus.current_job.job_type === 'optimize_only' ? 'Optimization' : 'NPT Simulation'}<br/>
                          Model: {systemStatus.current_job.model}
                        </span>
                        
                        {(systemStatus.current_job.progress !== undefined || systemStatus.current_job.status_message || systemStatus.current_job.eta) && (
                          <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.15)' }}>
                            {systemStatus.current_job.status_message && (
                              <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '6px', color: '#FBBF24' }}>
                                {systemStatus.current_job.status_message}
                              </div>
                            )}
                            {systemStatus.current_job.progress !== undefined && systemStatus.current_job.progress > 0 && (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px' }}>
                                <div style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.2)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                                  <div style={{ width: `${(systemStatus.current_job.progress * 100).toFixed(1)}%`, backgroundColor: '#10B981', height: '100%' }}></div>
                                </div>
                                <span style={{ fontSize: '11px', fontWeight: 'bold' }}>{(systemStatus.current_job.progress * 100).toFixed(0)}%</span>
                              </div>
                            )}
                            {systemStatus.current_job.eta && (
                              <div style={{ fontSize: '11px', opacity: 0.9, marginTop: '6px' }}>
                                ⏱️ ETA: <strong style={{ color: '#FBBF24' }}>{systemStatus.current_job.eta}</strong>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <button className="btn btn-danger" onClick={handleCancelJob}>
                        <StopCircle size={16} /> Stop Current Job
                      </button>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>
                      No active job running.
                    </div>
                  )}
                </div>

                {/* Queue Manager */}
                <div className="dashboard-card">
                  <h3><Layers size={18} style={{ color: 'var(--color-primary)' }} /> Job Queue Manager</h3>
                  <div style={{ overflowX: 'auto', maxHeight: '200px' }}>
                    {systemStatus.queue.length > 0 ? (
                      <table className="job-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Project</th>
                            <th>Model</th>
                            <th>Type</th>
                            <th>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {systemStatus.queue.map((job, idx) => (
                            <tr key={idx}>
                              <td>{idx}</td>
                              <td>{job.project_name}</td>
                              <td>{job.model}</td>
                              <td>{job.job_type === 'optimize_only' ? 'OPT' : 'NPT'}</td>
                              <td>
                                <button className="btn btn-secondary" style={{ padding: '4px 8px', width: 'auto' }} 
                                  onClick={() => handleDeleteQueueJob(idx)}>
                                  <Trash2 size={12} style={{ color: 'var(--color-danger)' }} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px 0' }}>
                        The queue is currently empty.
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* Real-time Monitoring Graphs */}
              <div className="dashboard-card" style={{ gap: '20px' }}>
                <h3><Activity size={18} style={{ color: 'var(--color-info)' }} /> Live Monitoring</h3>
                
                {systemStatus.current_job && systemStatus.current_job.job_type === 'optimize_only' ? (
                  optRealtimeData.length > 0 ? (
                    <div className="monitoring-section" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div className="alert alert-success" style={{ margin: 0, padding: '8px 12px', fontSize: '13px' }}>
                        📈 Structure Optimization Progress (FIRE Algorithm)
                      </div>
                      
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                        <div className="chart-container" style={{ padding: '16px', backgroundColor: 'rgba(0,0,0,0.15)', borderRadius: '8px' }}>
                          <h4 style={{ marginBottom: '8px', fontSize: '13px', textAlign: 'center' }}>Maximum Force Norm (eV/Å)</h4>
                          {(() => {
                            const data = optRealtimeData
                            const maxFmax = Math.max(...data.map(d => d.fmax))
                            const minFmax = Math.min(...data.map(d => d.fmax))
                            const range = maxFmax - minFmax || 1
                            const points = data.map((d, index) => {
                              const x = (index / (data.length - 1 || 1)) * 100
                              const y = 100 - ((d.fmax - minFmax) / range) * 100
                              return `${x},${y}`
                            }).join(' ')
                            return (
                              <div style={{ position: 'relative', height: '140px', width: '100%', borderBottom: '1px solid rgba(255,255,255,0.2)', borderLeft: '1px solid rgba(255,255,255,0.2)', boxSizing: 'border-box', marginTop: '12px' }}>
                                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
                                  <polyline fill="none" stroke="#FBBF24" strokeWidth="2" points={points} />
                                </svg>
                                <span style={{ position: 'absolute', top: 0, left: '4px', fontSize: '9px', color: 'var(--text-muted)' }}>Max: {maxFmax.toFixed(4)}</span>
                                <span style={{ position: 'absolute', bottom: 0, left: '4px', fontSize: '9px', color: 'var(--text-muted)' }}>Min: {minFmax.toFixed(4)}</span>
                                <span style={{ position: 'absolute', bottom: '-15px', right: 0, fontSize: '9px', color: 'var(--text-muted)' }}>Step {data[data.length-1].step}</span>
                              </div>
                            )
                          })()}
                        </div>
                        <div className="chart-container" style={{ padding: '16px', backgroundColor: 'rgba(0,0,0,0.15)', borderRadius: '8px' }}>
                          <h4 style={{ marginBottom: '8px', fontSize: '13px', textAlign: 'center' }}>Potential Energy (eV)</h4>
                          {(() => {
                            const data = optRealtimeData
                            const maxEnergy = Math.max(...data.map(d => d.energy))
                            const minEnergy = Math.min(...data.map(d => d.energy))
                            const range = maxEnergy - minEnergy || 1
                            const points = data.map((d, index) => {
                              const x = (index / (data.length - 1 || 1)) * 100
                              const y = 100 - ((d.energy - minEnergy) / range) * 100
                              return `${x},${y}`
                            }).join(' ')
                            return (
                              <div style={{ position: 'relative', height: '140px', width: '100%', borderBottom: '1px solid rgba(255,255,255,0.2)', borderLeft: '1px solid rgba(255,255,255,0.2)', boxSizing: 'border-box', marginTop: '12px' }}>
                                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
                                  <polyline fill="none" stroke="#6366F1" strokeWidth="2" points={points} />
                                </svg>
                                <span style={{ position: 'absolute', top: 0, left: '4px', fontSize: '9px', color: 'var(--text-muted)' }}>Max: {maxEnergy.toFixed(4)}</span>
                                <span style={{ position: 'absolute', bottom: 0, left: '4px', fontSize: '9px', color: 'var(--text-muted)' }}>Min: {minEnergy.toFixed(4)}</span>
                                <span style={{ position: 'absolute', bottom: '-15px', right: 0, fontSize: '9px', color: 'var(--text-muted)' }}>Step {data[data.length-1].step}</span>
                              </div>
                            )
                          })()}
                        </div>
                      </div>

                      <div style={{ maxHeight: '150px', overflowY: 'auto', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', fontSize: '12px', marginTop: '10px' }}>
                        <table className="job-table" style={{ margin: 0 }}>
                          <thead>
                            <tr>
                              <th>Step</th>
                              <th>Potential Energy (eV)</th>
                              <th>Max Force Norm (eV/Å)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {[...optRealtimeData].reverse().slice(0, 10).map((row, idx) => (
                              <tr key={idx}>
                                <td>{row.step}</td>
                                <td>{row.energy.toFixed(6)}</td>
                                <td style={{ color: row.fmax < 0.01 ? 'var(--color-success)' : 'inherit' }}>
                                  {row.fmax.toFixed(6)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '48px 0' }}>
                      <RefreshCw size={24} className="pulse" style={{ margin: '0 auto 12px auto' }} />
                      Waiting for structure optimization logs...
                    </div>
                  )
                ) : systemStatus.current_job && systemStatus.current_job.job_type !== 'optimize_only' ? (
                  realtimeData.length > 0 ? (
                    <div className="monitoring-section">
                      
                      {/* Lattice Lengths Chart (a, b, c) */}
                      <div className="chart-container">
                        <h4 style={{ marginBottom: '12px', fontSize: '13px', textAlign: 'center' }}>Lattice Parameters (a, b, c) - Å</h4>
                        <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px', justifyContent: 'center' }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#6366F1' }}></span> a</span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10B981' }}></span> b</span>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#EF4444' }}></span> c</span>
                        </div>
                        {(() => {
                          const chartData = getChartDataWithTemp()
                          const steps = chartData.length
                          const latticeVals = chartData.flatMap(d => [d.a_lengths, d.b_lengths, d.c_lengths])
                          const minLat = Math.min(...latticeVals)
                          const maxLat = Math.max(...latticeVals)
                          const rangeLat = maxLat - minLat || 1
                          const yMinLat = minLat - rangeLat * 0.05
                          const yMaxLat = maxLat + rangeLat * 0.05
                          const getPoints = (key) => {
                            return chartData.map((d) => {
                              const x = 50 + ((d.calculatedTemp - minTemp) / rangeTemp) * 430
                              const y = 180 - ((d[key] - yMinLat) / yRangeLat) * 160
                              return `${x},${y}`
                            }).join(' ')
                          }

                          return (
                            <svg className="chart-svg" viewBox="0 0 500 210" style={{ overflow: 'visible' }}>
                              {/* Y grid lines */}
                              <line x1="50" y1="20" x2="480" y2="20" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="73" x2="480" y2="73" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="126" x2="480" y2="126" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="180" x2="480" y2="180" stroke="rgba(255,255,255,0.1)" />
                              {/* X Axes */}
                              <line x1="50" y1="180" x2="480" y2="180" stroke="rgba(255,255,255,0.1)" />
                              <line x1="50" y1="20" x2="50" y2="180" stroke="rgba(255,255,255,0.1)" />
                              <line x1="265" y1="20" x2="265" y2="180" stroke="rgba(255,255,255,0.03)" />
                              <line x1="480" y1="20" x2="480" y2="180" stroke="rgba(255,255,255,0.05)" />
                              
                              {/* Y-Axis Tick Labels */}
                              <text x="42" y="24" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{yMaxLat.toFixed(2)}</text>
                              <text x="42" y="77" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{(yMaxLat - yRangeLat * 1/3).toFixed(2)}</text>
                              <text x="42" y="130" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{(yMaxLat - yRangeLat * 2/3).toFixed(2)}</text>
                              <text x="42" y="184" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{yMinLat.toFixed(2)}</text>

                              {/* X-Axis Tick Labels */}
                              <text x="50" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{minTemp.toFixed(0)} K</text>
                              <text x="265" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{((minTemp + maxTemp) / 2).toFixed(0)} K</text>
                              <text x="480" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{maxTemp.toFixed(0)} K</text>

                              {/* Plot lines */}
                              <polyline points={getPoints('a_lengths')} stroke="#6366F1" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                              <polyline points={getPoints('b_lengths')} stroke="#10B981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                              <polyline points={getPoints('c_lengths')} stroke="#EF4444" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                            </svg>
                          )
                        })()}
                      </div>

                      {/* Volumes Chart */}
                      <div className="chart-container">
                        <h4 style={{ marginBottom: '12px', fontSize: '13px', textAlign: 'center' }}>Volume - Å³</h4>
                        <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px', justifyContent: 'center' }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}><span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10B981' }}></span> Volume</span>
                        </div>
                        {/* Custom SVG Line Chart */}
                        {(() => {
                          const chartData = getChartDataWithTemp()
                          const steps = chartData.length
                          const volVals = chartData.map(d => d.volumes)
                          const minVol = Math.min(...volVals)
                          const maxVol = Math.max(...volVals)
                          const rangeVol = maxVol - minVol || 1
                          const yMinVol = minVol - rangeVol * 0.05
                          const yMaxVol = maxVol + rangeVol * 0.05
                          const yRangeVol = yMaxVol - yMinVol || 1

                          const minTemp = chartData[0]?.calculatedTemp || 0
                          const maxTemp = chartData[steps - 1]?.calculatedTemp || 100
                          const rangeTemp = maxTemp - minTemp || 1

                          const getPoints = () => {
                            return chartData.map((d) => {
                              const x = 50 + ((d.calculatedTemp - minTemp) / rangeTemp) * 430
                              const y = 180 - ((d.volumes - yMinVol) / yRangeVol) * 160
                              return `${x},${y}`
                            }).join(' ')
                          }

                          return (
                            <svg className="chart-svg" viewBox="0 0 500 210" style={{ overflow: 'visible' }}>
                              {/* Y grid lines */}
                              <line x1="50" y1="20" x2="480" y2="20" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="73" x2="480" y2="73" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="126" x2="480" y2="126" stroke="rgba(255,255,255,0.05)" />
                              <line x1="50" y1="180" x2="480" y2="180" stroke="rgba(255,255,255,0.1)" />
                              {/* X Axes */}
                              <line x1="50" y1="180" x2="480" y2="180" stroke="rgba(255,255,255,0.1)" />
                              <line x1="50" y1="20" x2="50" y2="180" stroke="rgba(255,255,255,0.1)" />
                              <line x1="265" y1="20" x2="265" y2="180" stroke="rgba(255,255,255,0.03)" />
                              <line x1="480" y1="20" x2="480" y2="180" stroke="rgba(255,255,255,0.05)" />
                              
                              {/* Y-Axis Tick Labels */}
                              <text x="42" y="24" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{yMaxVol.toFixed(1)}</text>
                              <text x="42" y="77" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{(yMaxVol - yRangeVol * 1/3).toFixed(1)}</text>
                              <text x="42" y="130" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{(yMaxVol - yRangeVol * 2/3).toFixed(1)}</text>
                              <text x="42" y="184" textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{yMinVol.toFixed(1)}</text>

                              {/* X-Axis Tick Labels */}
                              <text x="50" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{minTemp.toFixed(0)} K</text>
                              <text x="265" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{((minTemp + maxTemp) / 2).toFixed(0)} K</text>
                              <text x="480" y="196" textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="monospace">{maxTemp.toFixed(0)} K</text>

                              {/* Plot lines */}
                              <polyline points={getPoints()} stroke="#10B981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                            </svg>
                          )
                        })()}
                      </div>

                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '48px 0' }}>
                      <RefreshCw size={24} className="pulse" style={{ margin: '0 auto 12px auto' }} />
                      Waiting for first batch to complete...
                    </div>
                  )
                ) : (
                  <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '48px 0' }}>
                    Active NPT simulation running is required to show charts.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 2. Optimizer Tab */}
          {activeTab === 'optimizer' && (
            <div className="dashboard-card" style={{ maxWidth: '720px', margin: '0 auto' }}>
              <h3>🔬 Structure Optimizer</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginBottom: '16px' }}>
                Upload a structure CIF file. The optimizer will perform relaxation (FIRE algorithm) using the selected Force Field (CHGNet) and yield the optimized structure.
              </p>
              
              <div 
                className={`file-uploader ${optDragActive ? 'dragover' : ''} ${optFiles.length === 0 ? 'invalid' : ''}`}
                onDragEnter={e => handleDrag(e, setOptDragActive)}
                onDragOver={e => handleDrag(e, setOptDragActive)}
                onDragLeave={e => handleDrag(e, setOptDragActive)}
                onDrop={e => handleDrop(e, setOptDragActive, setOptFiles, true)}
                onClick={() => optFileInputRef.current.click()}
              >
                <UploadCloud size={40} className={optFiles.length > 0 ? "active" : ""} />
                <p style={{ fontSize: '15px', fontWeight: 500 }}>Drag & drop CIF files here, or click to choose</p>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Max file size: 1080 atoms limit</span>
              </div>
              
              {optFiles.length > 0 && (
                <div className="uploaded-files-list">
                  {optFiles.map((file, idx) => (
                    <div key={idx} className="uploaded-file-item">
                      <span>{file.name} ({(file.size/1024).toFixed(1)} KB)</span>
                      <button type="button" className="btn btn-secondary" style={{ width: 'auto', padding: '4px 8px' }} 
                        onClick={(e) => { e.stopPropagation(); setOptFiles(prev => prev.filter((_, i) => i !== idx)); }}>
                        <Trash2 size={14} style={{ color: 'var(--color-danger)' }} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 3. Project Browser Tab */}
          {activeTab === 'projects' && (
            <div className="project-browser-layout">
              
              {/* Project Selection Sidebar */}
              <div className="project-sidebar">
                <h4 style={{ marginBottom: '16px', fontSize: '13px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Saved Projects</h4>
                {projects.length > 0 ? (
                  <div className="project-list">
                    {projects.map((p, idx) => (
                      <div key={idx} className={`project-item ${selectedProject === p ? 'active' : ''}`}
                        onClick={() => setSelectedProject(p)}>
                        {p}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '16px 0' }}>
                    No projects found.
                  </div>
                )}
              </div>

              {/* Project Details Panel */}
              <div className="project-details">
                {selectedProject && projectDetails ? (
                  <>
                    <div className="project-header-row">
                      <div>
                        <h2>{selectedProject}</h2>
                      </div>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="btn btn-secondary" style={{ width: 'auto' }} onClick={() => handleCreateZip(selectedProject)}>
                          📦 Create/Update ZIP Pack
                        </button>
                        <button className="btn btn-danger" style={{ width: 'auto' }} onClick={() => handleDeleteProject(selectedProject)}>
                          <Trash2 size={16} /> Delete Project
                        </button>
                      </div>
                    </div>

                    <div className="project-meta-grid">
                      <div className="meta-box">
                        <label>Calculation Time</label>
                        <value>{projectDetails.execution_time ? `${projectDetails.execution_time} seconds` : 'N/A'}</value>
                      </div>
                    </div>

                    {/* Plots Grid */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <h3>Plots & Visualizations</h3>
                      <div className="plots-grid">
                        
                        {/* Temperature-dependent summary plot */}
                        {projectDetails.plots && projectDetails.plots.includes('npt_vs_temp.png') && (
                          <div className="plot-card">
                            <h4>NPT VS Temperature Properties</h4>
                            <img src={`/api/projects/${selectedProject}/files/npt_vs_temp.png`} alt="NPT vs Temp" />
                          </div>
                        )}
                        
                        {/* RDF Analysis plot */}
                        {projectDetails.plots && projectDetails.plots.includes('RDF_Total.png') && (
                          <div className="plot-card">
                            <h4>Radial Distribution Function (RDF)</h4>
                            <img src={`/api/projects/${selectedProject}/files/RDF_Total.png`} alt="RDF Total" />
                          </div>
                        )}

                        {/* Dynamics Tilt Analysis plot */}
                        {projectDetails.plots && projectDetails.plots.includes('Dynamics_Tilt_Angle.png') && (
                          <div className="plot-card">
                            <h4>Dynamics Tilt Angle</h4>
                            <img src={`/api/projects/${selectedProject}/files/Dynamics_Tilt_Angle.png`} alt="Dynamics Tilt" />
                          </div>
                        )}

                        {/* Dynamics Bond Length plot */}
                        {projectDetails.plots && projectDetails.plots.includes('Dynamics_B-Site_Bond.png') && (
                          <div className="plot-card">
                            <h4>B-Site Bond Length</h4>
                            <img src={`/api/projects/${selectedProject}/files/Dynamics_B-Site_Bond.png`} alt="B-Site Bond" />
                          </div>
                        )}

                      </div>
                    </div>

                    {/* Downloadable Artifacts */}
                    <div className="download-section">
                      <h3>Download Artifact Files</h3>
                      <div className="download-grid">
                        
                        {projectDetails.zip_exists && (
                          <a href={`/api/projects/${selectedProject}/files/analysis_results.zip`} download className="btn btn-primary" style={{ textDecoration: 'none' }}>
                            <Download size={14} /> Download ZIP Archive
                          </a>
                        )}

                        {projectDetails.files && projectDetails.files.map((file, idx) => (
                          <a key={idx} href={`/api/projects/${selectedProject}/files/${file}`} download className="btn btn-secondary" style={{ textDecoration: 'none', fontSize: '12px' }}>
                            <Download size={12} /> {file}
                          </a>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)' }}>
                    Select a project from the sidebar list to inspect results.
                  </div>
                )}
              </div>

            </div>
          )}

          {/* 4. CIF Structure Editor Tab */}
          {activeTab === 'editor' && (
            <div className="cif-editor-grid">
              
              {/* Form Input Panel */}
              <div className="dashboard-card">
                <h3>🛠️ CIF Structure Editor</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '13.5px' }}>
                  Upload a structure file. This editor lets you randomly or sequentially substitute elements or create vacancies inside the cell.
                </p>

                <form onSubmit={handleCifEditSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
                  
                  <div className="form-group">
                    <label>Operation Type</label>
                    <div style={{ display: 'flex', gap: '16px', marginTop: '4px' }}>
                      <label className="form-checkbox">
                        <input type="radio" name="operation" checked={cifEditorParams.operation === 'Substitution'}
                          onChange={() => setCifEditorParams({...cifEditorParams, operation: 'Substitution'})} />
                        Substitution
                      </label>
                      <label className="form-checkbox">
                        <input type="radio" name="operation" checked={cifEditorParams.operation === 'Vacancy'}
                          onChange={() => setCifEditorParams({...cifEditorParams, operation: 'Vacancy'})} />
                        Vacancy
                      </label>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Target Element</label>
                      <input type="text" className="form-input" value={cifEditorParams.replaceFrom}
                        onChange={e => setCifEditorParams({...cifEditorParams, replaceFrom: e.target.value})} />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Replace With</label>
                      <input type="text" className="form-input" value={cifEditorParams.operation === 'Vacancy' ? '' : cifEditorParams.replaceTo}
                        disabled={cifEditorParams.operation === 'Vacancy'}
                        onChange={e => setCifEditorParams({...cifEditorParams, replaceTo: e.target.value})} />
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Percentage (%)</label>
                    <input type="number" className={`form-input ${cifEditorParams.percentage === "" ? "invalid" : ""}`} min="0" max="100" step="1" value={cifEditorParams.percentage}
                      onChange={e => handleEditorPercentageChange(e.target.value)} />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Selection Mode</label>
                      <select className="form-select" value={cifEditorParams.mode}
                        onChange={e => setCifEditorParams({...cifEditorParams, mode: e.target.value})}>
                        <option value="random">Random</option>
                        <option value="sequential">Sequential</option>
                      </select>
                    </div>
                  </div>

                   <div className="form-group">
                    <label>Select CIF File</label>
                    <div 
                      className={`compact-uploader ${editorDragActive ? 'dragover' : ''} ${!editorFile ? 'invalid' : ''}`}
                      onDragEnter={e => handleDrag(e, setEditorDragActive)}
                      onDragOver={e => handleDrag(e, setEditorDragActive)}
                      onDragLeave={e => handleDrag(e, setEditorDragActive)}
                      onDrop={e => handleDrop(e, setEditorDragActive, setEditorFile, false)}
                      onClick={() => editorFileInputRef.current.click()}
                    >
                      <UploadCloud size={20} className={editorFile ? "active" : ""} />
                      <span>Drag & drop CIF here</span>
                      <span className="subtext">or click to browse</span>
                      <input 
                        type="file" 
                        ref={editorFileInputRef} 
                        style={{ display: 'none' }} 
                        accept=".cif" 
                        onChange={e => setEditorFile(e.target.files[0])} 
                      />
                    </div>
                    {editorFile && (
                      <div className="uploaded-files-list-sidebar">
                        <div className="uploaded-file-item-sidebar">
                          <span>{editorFile.name} ({(editorFile.size/1024).toFixed(1)} KB)</span>
                          <button type="button" onClick={(e) => { e.stopPropagation(); setEditorFile(null); }}>×</button>
                        </div>
                      </div>
                    )}
                  </div>

                  <button type="submit" className="btn btn-primary" style={{ marginTop: '12px' }} disabled={cifEditorParams.percentage === "" || !editorFile}>
                    Run CIF Processor
                  </button>
                </form>
              </div>

              {/* Console Logs / Download Output Panel */}
              <div className="dashboard-card">
                <h3>Console Logs & Output</h3>
                
                <div className="log-console">
                  {editorLogs.length > 0 ? (
                    editorLogs.map((log, idx) => {
                      let logClass = 'log-entry-info'
                      if (log.includes('Error')) logClass = 'log-entry-error'
                      else if (log.includes('Success')) logClass = 'log-entry-success'
                      else if (log.includes('Warning')) logClass = 'log-entry-warning'
                      
                      return (
                        <div key={idx} className={logClass}>
                          &gt; {log}
                        </div>
                      )
                    })
                  ) : (
                    <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                      Console log is empty. Trigger CIF processing.
                    </div>
                  )}
                </div>

                {editedCifFile && (
                  <div className="alert alert-success" style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div>
                      <strong style={{ color: '#34D399' }}><CheckCircle2 size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} /> Success!</strong>
                      <p style={{ fontSize: '13px', marginTop: '4px', color: '#A7F3D0' }}>CIF structure modified successfully.</p>
                    </div>
                    <button className="btn btn-primary" onClick={() => downloadFile(editedCifFile.name, editedCifFile.content, "chemical/x-cif")}>
                      <Download size={16} /> Download '{editedCifFile.name}'
                    </button>
                  </div>
                )}
              </div>

            </div>
          )}

          {activeTab === 'logs' && (
            <div className="logs-dashboard">
              
              {/* Left Column: System Status Sidebar */}
              <div className="logs-sidebar">
                
                {/* 1. Supervisor Services */}
                <div className="dashboard-card">
                  <h3><Server size={18} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} /> Supervisor Services</h3>
                  <div className="status-grid">
                    {detailedStatus ? (
                      <pre className="pre-status">{detailedStatus.supervisor_status}</pre>
                    ) : (
                      <p className="loading-text">Loading Supervisor status...</p>
                    )}
                  </div>
                </div>

                {/* 2. CUDA & GPU Information */}
                <div className="dashboard-card">
                  <h3><Cpu size={18} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} /> PyTorch CUDA / GPU</h3>
                  {detailedStatus ? (
                    <div className="system-metrics">
                      <div className="metric-row">
                        <span className="metric-label">CUDA Available:</span>
                        <span className={`metric-value ${detailedStatus.cuda.available ? 'text-success' : 'text-danger'}`}>
                          {detailedStatus.cuda.available ? 'YES' : 'NO (CPU Fallback)'}
                        </span>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">Device Count:</span>
                        <span className="metric-value">{detailedStatus.cuda.device_count}</span>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">PyTorch Version:</span>
                        <span className="metric-value">{detailedStatus.cuda.pytorch_version}</span>
                      </div>
                      {detailedStatus.cuda.devices.map((dev, idx) => (
                        <div key={idx} className="gpu-device-card" style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid var(--border-color)' }}>
                          <div className="metric-row">
                            <span className="metric-label">GPU {dev.id}:</span>
                            <span className="metric-value font-semibold text-primary">{dev.name || "N/A"}</span>
                          </div>
                          {dev.capability && (
                            <div className="metric-row">
                              <span className="metric-label">Compute Capability:</span>
                              <span className="metric-value">{dev.capability.join('.')}</span>
                            </div>
                          )}
                          {dev.memory_allocated !== undefined && (
                            <div className="metric-row">
                              <span className="metric-label">VRAM Allocated:</span>
                              <span className="metric-value">{dev.memory_allocated.toFixed(1)} MB</span>
                            </div>
                          )}
                          {dev.error && (
                            <div className="metric-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                              <span className="metric-label text-danger">Error:</span>
                              <span className="metric-value text-danger" style={{ fontSize: '11px', whiteSpace: 'pre-wrap' }}>{dev.error}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="loading-text">Loading GPU status...</p>
                  )}
                </div>

                {/* 3. Container Hardware Stats */}
                <div className="dashboard-card">
                  <h3>Resource Usage</h3>
                  {detailedStatus ? (
                    <div className="system-metrics">
                      <div className="metric-row">
                        <span className="metric-label">Available CPU Cores:</span>
                        <span className="metric-value">{detailedStatus.cpu_count}</span>
                      </div>
                      <div className="metric-row" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                        <span className="metric-label">Memory Info (RAM):</span>
                        <pre className="pre-status" style={{ fontSize: '11px', marginTop: '6px' }}>{detailedStatus.ram_info}</pre>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">Disk Total:</span>
                        <span className="metric-value">{detailedStatus.disk.total_gb} GB</span>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">Disk Used:</span>
                        <span className="metric-value">{detailedStatus.disk.used_gb} GB ({Math.round(detailedStatus.disk.used_gb / detailedStatus.disk.total_gb * 100) || 0}%)</span>
                      </div>
                      <div className="metric-row">
                        <span className="metric-label">Disk Free:</span>
                        <span className="metric-value text-success">{detailedStatus.disk.free_gb} GB</span>
                      </div>
                    </div>
                  ) : (
                    <p className="loading-text">Loading Hardware stats...</p>
                  )}
                </div>
                
              </div>

              {/* Right Column: Console Log Viewer */}
              <div className="logs-main">
                <div className="dashboard-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <div className="logs-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px', borderBottom: '1px solid var(--border-color)', paddingBottom: '15px', marginBottom: '15px' }}>
                    <h3 style={{ margin: 0 }}><Terminal size={18} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} /> Container Log Console</h3>
                    
                    <div className="logs-controls" style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                      {/* Log Type Select */}
                      <select 
                        className="form-input" 
                        value={logType} 
                        onChange={(e) => {
                          setLogType(e.target.value)
                          fetchSystemLogs(e.target.value, tailLines)
                        }}
                        style={{ width: 'auto', padding: '6px 12px' }}
                      >
                        <option value="worker">Worker Internal Log</option>
                        <option value="api_stdout">FastAPI Server stdout</option>
                        <option value="api_stderr">FastAPI Server stderr</option>
                        <option value="jupyter_stdout">Jupyter Lab stdout</option>
                        <option value="jupyter_stderr">Jupyter Lab stderr</option>
                        <option value="supervisord">Supervisord Log</option>
                      </select>

                      {/* Tail Lines Select */}
                      <select
                        className="form-input"
                        value={tailLines}
                        onChange={(e) => {
                          const lines = Number(e.target.value)
                          setTailLines(lines)
                          fetchSystemLogs(logType, lines)
                        }}
                        style={{ width: 'auto', padding: '6px 12px' }}
                      >
                        <option value="100">Last 100 lines</option>
                        <option value="500">Last 500 lines</option>
                        <option value="1000">Last 1000 lines</option>
                        <option value="2000">Last 2000 lines</option>
                      </select>

                      {/* Auto Refresh */}
                      <label className="checkbox-label" style={{ fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)' }}>
                        <input 
                          type="checkbox" 
                          checked={autoRefreshLogs} 
                          onChange={(e) => setAutoRefreshLogs(e.target.checked)} 
                        />
                        Auto Refresh (5s)
                      </label>

                      {/* Refresh Now */}
                      <button className="btn btn-secondary" onClick={() => { fetchSystemLogs(logType, tailLines); fetchDetailedStatus(); }} disabled={isLogsLoading} style={{ padding: '6px 12px' }}>
                        <RefreshCw size={14} className={isLogsLoading ? 'spin' : ''} />
                      </button>

                      {/* Clear Log */}
                      <button className="btn btn-danger" onClick={() => clearSystemLog(logType)} style={{ padding: '6px 12px' }}>
                        <Trash2 size={14} /> Clear Log
                      </button>

                      {/* Download */}
                      <a 
                        className="btn btn-primary" 
                        href={`/api/system/logs/${logType}?download=true`} 
                        download
                        style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px' }}
                      >
                        <Download size={14} /> Download
                      </a>
                    </div>
                  </div>

                  {/* Log Console Box */}
                  <div className="log-console-box" style={{ flex: 1, minHeight: '350px', background: '#0F172A', border: '1px solid #1E293B', borderRadius: '8px', padding: '15px', overflow: 'auto', maxHeight: '550px' }}>
                    <pre className="console-text" style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontFamily: 'Courier New, Courier, monospace', fontSize: '12px', color: '#E2E8F0', textAlign: 'left' }}>{logContent || "No logs loaded."}</pre>
                  </div>

                  {/* Process List (ps aux) & nvidia-smi */}
                  <div className="logs-extra-grid" style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    <div className="logs-extra-section" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <h4 style={{ margin: 0, color: 'var(--text-color)' }}>Active Container Processes (ps aux)</h4>
                      <pre className="pre-extra-status" style={{ flex: 1, maxHeight: '200px', background: '#1E293B', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '10px', overflow: 'auto', fontSize: '11px', color: '#94A3B8', fontFamily: 'monospace', margin: 0, textAlign: 'left' }}>{detailedStatus?.ps_aux || "Loading..."}</pre>
                    </div>
                    <div className="logs-extra-section" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <h4 style={{ margin: 0, color: 'var(--text-color)' }}>System GPU Status (nvidia-smi)</h4>
                      <pre className="pre-extra-status" style={{ flex: 1, maxHeight: '200px', background: '#1E293B', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '10px', overflow: 'auto', fontSize: '11px', color: '#94A3B8', fontFamily: 'monospace', margin: 0, textAlign: 'left' }}>{detailedStatus?.nvidia_smi || "Loading..."}</pre>
                    </div>
                  </div>
                  
                </div>
              </div>
              
            </div>
          )}

        </div>
      </main>

    </div>
  )
}
