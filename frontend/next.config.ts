
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://taspol-ai-service-dev.hf.space/api/:path*' // FastAPI server
      }
    ]
  }
}

module.exports = nextConfig
