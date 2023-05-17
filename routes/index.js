const router = require('express').Router()

router.get('/', async(req, res)=>{
    return res.json({success: true, message: 'Welcome to backend.'})
})

router.get('/models', require('./listModels'))
router.post('/predict', require('./predictOutput'))
router.post('/track', require('./trackUser'))

module.exports = router