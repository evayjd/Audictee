<script setup>
import { ref, watch, nextTick, onUnmounted } from "vue"
import axios from "axios"

// --- Données réactives (响应式数据) ---
const url = ref("")
const videoId = ref("")
const sentences = ref([])
const loading = ref(false)
const activeIndex = ref(-1)
const sentenceRefs = ref([]) 

let interval = null
let player = null

// --- Extraire l'ID de la vidéo (解析 ID) ---
const extractVideoId = (url) => {
  const match = url.match(/(?:v=|\/)([0-9A-Za-z_-]{11})/)
  return match ? match[1] : ""
}

// --- Appeler l'API pour obtenir la transcription (获取字幕) ---
const fetchTranscript = async () => {
  if (!url.value) return

  loading.value = true
  sentences.value = []
  videoId.value = extractVideoId(url.value)

  try {
    const res = await axios.post("http://localhost:8000/api/transcript", {
      url: url.value,
    })
    // 确保后端返回的数据结构包含 sentences 且每个 sentence 有 tokens
    sentences.value = res.data.sentences 
  } catch (err) {
    console.error("Erreur API :", err)
    alert("Impossible de récupérer la transcription. Veuillez vérifier le serveur backend.")
  } finally {
    loading.value = false
  }
}

// --- Contrôle de YouTube ---
const seekTo = (time) => {
  if (player && player.seekTo) {
    player.seekTo(time, true)
  }
}

const loadYouTubeAPI = () => {
  return new Promise((resolve) => {
    if (window.YT && window.YT.Player) {
      resolve()
    } else {
      const tag = document.createElement("script")
      tag.src = "https://www.youtube.com/iframe_api"
      document.body.appendChild(tag)
      window.onYouTubeIframeAPIReady = () => resolve()
    }
  })
}

watch(videoId, async (newId) => {
  if (!newId) return
  await loadYouTubeAPI()

  if (player) {
    player.loadVideoById(newId)
  } else {
    player = new window.YT.Player("player", {
      height: "360",
      width: "100%",
      videoId: newId,
      events: {
        onReady: () => startTracking(),
        onStateChange: (event) => {
          if (event.data === window.YT.PlayerState.PLAYING) {
            startTracking()
          } else {
            if (interval) clearInterval(interval)
          }
        }
      }
    })
  }
})

const startTracking = () => {
  if (interval) clearInterval(interval)
  interval = setInterval(() => {
    if (!player || !player.getCurrentTime) return
    const currentTime = player.getCurrentTime()
    
    const index = sentences.value.findIndex(s => 
      currentTime >= s.start && currentTime <= s.end
    )
    
    if (index !== -1 && index !== activeIndex.value) {
      activeIndex.value = index
    }
  }, 250)
}

watch(activeIndex, async (newIndex) => {
  await nextTick()
  const el = sentenceRefs.value[newIndex]
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" })
  }
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})
</script>

<template>
  <div style="max-width: 900px; margin: 0 auto; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <h1 style="text-align: center; color: #2d3436;">Lecteur de Langue IA</h1>

    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
      <input
        v-model="url"
        placeholder="Collez le lien de la vidéo YouTube ici..."
        style="flex: 1; padding: 12px; border: 2px solid #dfe6e9; border-radius: 8px; outline: none;"
      />
      <button 
        @click="fetchTranscript" 
        :disabled="loading"
        style="padding: 0 24px; background: #0984e3; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;"
      >
        {{ loading ? 'Chargement...' : 'Charger' }}
      </button>
    </div>

    <div v-show="videoId" style="border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
      <div id="player"></div>
    </div>

    <div 
      v-if="sentences.length > 0"
      style="
      margin-top: 25px; 
      max-height: 450px; 
      overflow-y: 
      auto; border: 1px solid #eee; 
      border-radius: 12px; 
      padding: 15px; 
      background: #fff;
      max-width: 700px;
      margin-left: auto
      margin-right: auto;
      "
    >
      <div
        v-for="(s, index) in sentences"
        :key="index"
        :ref="el => { if (el) sentenceRefs[index] = el }"
        @click="seekTo(s.start)"
        :style="{
          padding: '14px 18px',
          margin: '12px 0',
          cursor: 'pointer',
          borderRadius: '10px',
          transition: 'all 0.2s ease',
          backgroundColor: index === activeIndex ? '#fff9db' : 'transparent',
          borderLeft: index === activeIndex ? '4px solid #fcc419' : '4px solid transparent',
          color: index === activeIndex ? '#000' : '#444',
          fontSize: '18px',
          lineHeight: '1.8',
          wordBreak: 'normal',
          overflowWrap: 'break-word',
          whiteSpace:'normal',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems:'baseline',
          columnGap: '0px'
        }"
      >
        <span
          v-for="(token, tIndex) in s.tokens"
          :key="tIndex"
          :style="{
            marginRight: '4px',
            fontWeight: token.pos === 'VERB' ? 'bold' : 'normal',
            color:
              token.pos === 'NOUN'
                ? '#a8c4e6'
                : token.pos === 'VERB'
                ? '#f4a4a4'
                : token.pos === 'ADJ'
                ? '#d9c36c'
                : 'inherit' 
          }"
        >
          {{ token.text }}
        </span>
      </div>
    </div>
    
    <div v-else-if="!loading" style="text-align: center; color: #b2bec3; margin-top: 50px;">
      En attente d'un lien pour charger la transcription...
    </div>
  </div>
</template>