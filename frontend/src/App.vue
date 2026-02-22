<script setup>
import { ref, watch, nextTick, onMounted, onUnmounted } from "vue"
import axios from "axios"

const url = ref("")
const videoId = ref("")
const sentences = ref([])
const loading = ref(false)
const activeIndex = ref(-1)
const sentenceRefs = ref([])
const grammarMode = ref(true) // true: Grammar(tokens) / false: Reading(text)

// ---- Fill-blank mode ----
const fillBlankMode = ref(false)
const BLANK_RATIO = 0.3
// blankMask[sIndex][tIndex] = true/false
const blankMask = ref({})
// answers[sIndex][tIndex] = { value, checked, correct }
const answers = ref({})

let interval = null
let player = null

// ------- Popup (token info) -------
const popup = ref({
  visible: false,
  x: 0,
  y: 0,
  token: null,
})

const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

const showPopup = async (token, evt) => {
  if (!grammarMode.value) return
  // token click should NOT trigger seekTo / outer click
  evt?.stopPropagation?.()
  evt?.preventDefault?.()

  popup.value.token = token

  const margin = 12
  const approxW = 260
  const approxH = 140
  const x = clamp(evt.clientX + 10, margin, window.innerWidth - approxW - margin)
  const y = clamp(evt.clientY + 10, margin, window.innerHeight - approxH - margin)

  popup.value.x = x
  popup.value.y = y
  popup.value.visible = true

  await nextTick()
}

const hidePopup = () => {
  popup.value.visible = false
  popup.value.token = null
}

const onGlobalClick = () => {
  // any click not stopped closes popup
  if (popup.value.visible) hidePopup()
}

const onPopupClick = (evt) => {
  // clicking inside popup should not close it
  evt.stopPropagation()
}

// ------- Helpers -------
const extractVideoId = (inputUrl) => {
  const match = inputUrl.match(/(?:v=|\/)([0-9A-Za-z_-]{11})/)
  return match ? match[1] : ""
}

// Prefer backend tokens; fallback to naive split
const getTokens = (s) => {
  if (Array.isArray(s?.tokens) && s.tokens.length) return s.tokens
  const text = String(s?.text ?? "")
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => ({ text: w, lemma: w, pos: null }))
}

const isPunct = (t) => /^[,.;:!?%)\]}»]$/.test(String(t ?? ""))
const isOpenPunct = (t) => /^[([{«]$/.test(String(t ?? ""))

// ---- Fill blank rules ----
const shouldBlankCandidate = (token) => token?.pos === "NOUN" || token?.pos === "VERB"
const normalize = (s) => String(s ?? "").trim().toLowerCase()

const ensureAnswerSlot = (sIndex, tIndex) => {
  if (!answers.value[sIndex]) answers.value[sIndex] = {}
  if (!answers.value[sIndex][tIndex]) {
    answers.value[sIndex][tIndex] = { value: "", checked: false, correct: false }
  }
  return answers.value[sIndex][tIndex]
}

const onInput = (sIndex, tIndex, val) => {
  const slot = ensureAnswerSlot(sIndex, tIndex)
  slot.value = val
  slot.checked = false
  slot.correct = false
}

const checkAnswer = (sIndex, tIndex, token) => {
  const slot = ensureAnswerSlot(sIndex, tIndex)
  const user = normalize(slot.value)
  const gold = normalize(token?.text) // 如果想用 lemma 判答案：改为 token?.lemma
  slot.checked = true
  slot.correct = user.length > 0 && user === gold
}

const revealAnswer = (sIndex, tIndex, token) => {
  const slot = ensureAnswerSlot(sIndex, tIndex)
  slot.value = token?.text ?? ""
  slot.checked = true
  slot.correct = true
}

const resetBlankState = () => {
  blankMask.value = {}
  answers.value = {}
}

const generateBlankMask = () => {
  blankMask.value = {}

  sentences.value.forEach((s, sIndex) => {
    const tokens = getTokens(s)
    const candidates = tokens
      .map((t, i) => (shouldBlankCandidate(t) ? i : -1))
      .filter((i) => i !== -1)

    if (!candidates.length) return

    const blankCount = Math.max(1, Math.floor(candidates.length * BLANK_RATIO))
    const shuffled = [...candidates].sort(() => Math.random() - 0.5)

    blankMask.value[sIndex] = {}
    shuffled.slice(0, blankCount).forEach((i) => {
      blankMask.value[sIndex][i] = true
    })
  })
}

// ------- API -------
const fetchTranscript = async () => {
  if (!url.value) return

  loading.value = true
  sentences.value = []
  activeIndex.value = -1
  resetBlankState()
  hidePopup()
  videoId.value = extractVideoId(url.value)

  try {
    const res = await axios.post("http://localhost:8000/api/transcript", {
      url: url.value,
    })
    sentences.value = Array.isArray(res.data?.sentences) ? res.data.sentences : []
    generateBlankMask()
  } catch (err) {
    console.error("Erreur API :", err)
    alert("Impossible de récupérer la transcription. Veuillez vérifier le serveur backend.")
  } finally {
    loading.value = false
  }
}

// ------- YouTube -------
const seekTo = (time) => {
  if (player && player.seekTo) {
    player.seekTo(time, true)
  }
}

const loadYouTubeAPI = () => {
  return new Promise((resolve) => {
    if (window.YT && window.YT.Player) {
      resolve()
      return
    }
    const tag = document.createElement("script")
    tag.src = "https://www.youtube.com/iframe_api"
    document.body.appendChild(tag)
    window.onYouTubeIframeAPIReady = () => resolve()
  })
}

const startTracking = () => {
  if (interval) clearInterval(interval)
  interval = setInterval(() => {
    if (!player || !player.getCurrentTime) return
    const currentTime = player.getCurrentTime()

    const index = sentences.value.findIndex(
      (s) => currentTime >= s.start && currentTime <= s.end
    )

    if (index !== -1 && index !== activeIndex.value) {
      activeIndex.value = index
    }
  }, 250)
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
        },
      },
    })
  }
})

watch(activeIndex, async (newIndex) => {
  if (newIndex < 0) return
  await nextTick()
  const el = sentenceRefs.value[newIndex]
  if (el && el.scrollIntoView) {
    el.scrollIntoView({ behavior: "smooth", block: "center" })
  }
})

onMounted(() => {
  document.addEventListener("click", onGlobalClick, true)
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
  document.removeEventListener("click", onGlobalClick, true)
})
</script>

<template>
  <div
    @click="hidePopup"
    style="
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      position: relative;
    "
  >
    <h1 style="text-align: center; color: #2d3436;">Lecteur de Langue IA</h1>

    <div style="display: flex; gap: 10px; margin-bottom: 20px; align-items: center;">
      <input
        v-model="url"
        placeholder="Collez le lien de la vidéo YouTube ici..."
        style="
          flex: 1;
          padding: 12px;
          border: 2px solid #dfe6e9;
          border-radius: 8px;
          outline: none;
        "
      />

      <button
        @click.stop="fetchTranscript"
        :disabled="loading"
        style="
          padding: 0 24px;
          height: 42px;
          background: #83accc;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: bold;
        "
      >
        {{ loading ? "Chargement..." : "Charger" }}
      </button>

      <button
        @click.stop="
          grammarMode = !grammarMode;
          fillBlankMode = false;
          hidePopup();
        "
        style="
          padding: 0 16px;
          height: 42px;
          background: #945757;
          color: white;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-weight: bold;
        "
      >
        {{ grammarMode ? "Grammar mode" : "Reading mode" }}
      </button>

      <button
        @click.stop="
          fillBlankMode = !fillBlankMode;
          hidePopup();
          if (fillBlankMode) { if (!Object.keys(blankMask).length) generateBlankMask(); }
        "
        :disabled="!grammarMode || sentences.length === 0"
        :style="{
          padding: '0 16px',
          height: '42px',
          background: !grammarMode || sentences.length === 0 ? '#ccc' : (fillBlankMode ? '#2f855a' : '#2b6cb0'),
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: !grammarMode || sentences.length === 0 ? 'not-allowed' : 'pointer',
          fontWeight: 'bold',
          opacity: !grammarMode || sentences.length === 0 ? 0.6 : 1
        }"
        title="Only works in Grammar mode"
      >
        {{ fillBlankMode ? "Fill-blank: ON" : "Fill-blank: OFF" }}
      </button>
    </div>

    <div
      v-show="videoId"
      style="
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
      "
    >
      <div id="player"></div>
    </div>

    <div
      v-if="sentences.length > 0"
      style="
        margin-top: 25px;
        max-height: 450px;
        overflow-y: auto;
        border: 1px solid #eee;
        border-radius: 12px;
        padding: 15px;
        background: #fff;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
      "
    >
      <div
        v-for="(s, sIndex) in sentences"
        :key="sIndex"
        :ref="(el) => {
          if (el) sentenceRefs[sIndex] = el
        }"
        @click="seekTo(s.start)"
        :style="{
          padding: '14px 18px',
          margin: '12px 0',
          cursor: 'pointer',
          borderRadius: '10px',
          transition: 'all 0.2s ease',
          backgroundColor: sIndex === activeIndex ? '#fff9db' : 'transparent',
          borderLeft:
            sIndex === activeIndex ? '4px solid #fcc419' : '4px solid transparent',
          color: sIndex === activeIndex ? '#000' : '#444',
          fontSize: '18px',
          lineHeight: '1.8',
          wordBreak: 'normal',
          overflowWrap: 'break-word',
          whiteSpace: 'normal',
          textAlign: 'left',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'baseline',
          columnGap: '0px'
        }"
      >
        <!-- Grammar mode: token layout (supports fill-blank + popup) -->
        <template v-if="grammarMode">
          <template v-for="(token, tIndex) in getTokens(s)" :key="tIndex">
            <!-- fill blank input: only for selected 30% NOUN/VERB -->
            <template v-if="fillBlankMode && blankMask[sIndex]?.[tIndex]">
              <input
                :value="answers[sIndex]?.[tIndex]?.value ?? ''"
                @click.stop
                @input.stop="onInput(sIndex, tIndex, $event.target.value)"
                @keydown.enter.stop.prevent="checkAnswer(sIndex, tIndex, token)"
                @blur.stop="checkAnswer(sIndex, tIndex, token)"
                :placeholder="token.pos"
                :style="{
                  width: '110px',
                  maxWidth: '180px',
                  padding: '3px 8px',
                  borderRadius: '8px',
                  outline: 'none',
                  marginRight: isPunct(token.text) ? '0px' : '6px',
                  border: answers[sIndex]?.[tIndex]?.checked
                    ? (answers[sIndex]?.[tIndex]?.correct ? '2px solid #2f855a' : '2px solid #c53030')
                    : '2px solid #cbd5e0',
                  background: answers[sIndex]?.[tIndex]?.checked
                    ? (answers[sIndex]?.[tIndex]?.correct ? 'rgba(47,133,90,0.10)' : 'rgba(197,48,48,0.08)')
                    : '#fff',
                  fontSize: '16px',
                  lineHeight: '1.4',
                }"
              />
              <span
                @click.stop="revealAnswer(sIndex, tIndex, token)"
                style="font-size: 12px; color: #888; cursor: pointer; margin-right: 8px;"
                title="Reveal answer"
              >
                ↩
              </span>
            </template>

            <!-- normal token -->
            <span
              v-else
              @click.stop="showPopup(token, $event)"
              style="user-select: none;"
              :style="{
                marginRight: isPunct(token.text) ? '0px' : '4px',
                fontWeight: token.pos === 'VERB' ? 'bold' : 'normal',
                color:
                  token.pos === 'NOUN'
                    ? '#a8c4e6'
                    : token.pos === 'VERB'
                    ? '#f4a4a4'
                    : token.pos === 'ADJ'
                    ? '#d9c36c'
                    : 'inherit',
              }"
            >
              {{ token.text }}
            </span>

            <!-- optional: keep opening punctuation spacing simple -->
            <span v-if="false && isOpenPunct(token.text)"></span>
          </template>
        </template>

        <!-- ✅ Reading mode: pure text -->
        <span v-else>
          {{ s.text }}
        </span>
      </div>
    </div>

    <div
      v-else-if="!loading"
      style="text-align: center; color: #b2bec3; margin-top: 50px;"
    >
      En attente d'un lien pour charger la transcription...
    </div>

    <!-- Popup -->
    <div
      v-if="popup.visible && popup.token"
      @click="onPopupClick"
      :style="{
        position: 'fixed',
        left: popup.x + 'px',
        top: popup.y + 'px',
        zIndex: 9999,
        background: '#ffffff',
        border: '1px solid rgba(0,0,0,0.12)',
        borderRadius: '10px',
        padding: '10px 12px',
        boxShadow: '0 12px 30px rgba(0,0,0,0.15)',
        minWidth: '240px',
      }"
    >
      <div style="font-size: 13px; color: #666; margin-bottom: 6px;">
        Token info
      </div>

      <div style="display: grid; row-gap: 6px;">
        <div>
          <span style="font-size: 12px; color: #999;">text</span>
          <div style="font-size: 16px; font-weight: 700; color: #222;">
            {{ popup.token.text }}
          </div>
        </div>

        <div>
          <span style="font-size: 12px; color: #999;">lemma</span>
          <div style="font-size: 14px; color: #222;">
            {{ popup.token.lemma ?? "—" }}
          </div>
        </div>

        <div>
          <span style="font-size: 12px; color: #999;">POS</span>
          <div style="font-size: 14px; color: #222;">
            {{ popup.token.pos ?? "—" }}
          </div>
        </div>

        <div>
          <span style="font-size: 12px; color: #999;">Morph</span>
          <div style="font-size: 14px; color: #222;">
            <template v-if="popup.token.morph && Object.keys(popup.token.morph).length">
              <span
                v-for="(v, k) in popup.token.morph"
                :key="k"
                style="margin-right: 6px;"
              >
                {{ k }}={{ v }}
              </span>
            </template>
            <span v-else>—</span>
          </div>
        </div>

      </div>

      <div style="margin-top: 8px; font-size: 12px; color: #999;">
        click blank to close
      </div>
    </div>
  </div>
</template>